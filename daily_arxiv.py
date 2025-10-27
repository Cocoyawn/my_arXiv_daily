import os
import re
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import requests
import subprocess
import time

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# ======= Constants =======
# Hugging Face: Map arxiv_id to Hub spaces/models/datasets
HF_REPOS_API = "https://huggingface.co/api/arxiv/{arxiv_id}/repos"
HF_HEADERS = {"User-Agent": "arxiv-daily/1.0"}

# GitHub Search (Fallback)
GITHUB_SEARCH_REPO = "https://api.github.com/search/repositories"
GITHUB_SEARCH_CODE = "https://api.github.com/search/code"
GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "arxiv-daily/1.0"
}

# Prioritize the user-created PAT (MY_GITHUB_TOKEN) for higher rate limits
if os.getenv("MY_GITHUB_TOKEN"):
    logging.info("Using MY_GITHUB_TOKEN for GitHub API authentication.")
    GH_HEADERS["Authorization"] = f"Bearer {os.getenv('MY_GITHUB_TOKEN')}"
elif os.getenv("GITHUB_TOKEN"):
    # Fallback to the default token
    logging.info("Using default GITHUB_TOKEN. Warning: Search API rate limits may be low.")
    GH_HEADERS["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"
else:
    logging.warning("No GitHub API token found. Requests will be unauthenticated and likely rate-limited.")


# arXiv page
arxiv_url = "https://arxiv.org/"

# ======= Utility Functions =======

def load_config(config_file:str) -> dict:
    """
    Read the config and build the arXiv query string from keywords->filters:
    e.g., all:"Vision Language Model" OR all:"Vision-Language Model"
    """
    def pretty_filters(**config) -> dict:
        keywords = {}
        OR = ' OR '
        FIELD = 'all:'

        def quote_if_needed(s: str) -> str:
            s = s.strip()
            return f"\"{s}\"" if (' ' in s or '-' in s) else s

        def parse_filters(filters: list) -> str:
            terms = []
            for flt in filters:
                terms.append(FIELD + quote_if_needed(flt))
            return OR.join(terms)

        for k,v in config['keywords'].items():
            keywords[k] = parse_filters(v['filters'])
        return keywords

    with open(config_file,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config['kv'] = pretty_filters(**config)
        logging.info(f'config = {config}')
    return config

def get_authors(authors, first_author = False):
    if not authors:
        return ""
    if first_author:
        return str(authors[0])
    return ", ".join(str(author) for author in authors)

def sort_papers(papers):
    output = {}
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output

def http_get(url, headers=None, params=None, timeout=10, retries=2, sleep=0.8):
    """ Simple GET with retry mechanism """
    last_exc = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            else:
                logging.warning(f"GET {url} status={r.status_code} params={params} (Try {i+1}/{retries+1})")
                if r.status_code == 403:
                    logging.error(f"GitHub API Forbidden (403). Check your token and rate limits.")
        except Exception as e:
            last_exc = e
            logging.warning(f"GET {url} exception: {e} (Try {i+1}/{retries+1})")
        time.sleep(sleep)
    
    if last_exc:
        logging.error(f"Failed to GET {url} after {retries+1} attempts.")
    return None # Return None instead of raising an exception to avoid crashing the script

def get_code_link(qword:str) -> str | None:
    """
    Use GitHub repository search to find a possible implementation (sorted by stars).
    @param qword: Paper title or arxiv id
    @return Repository html_url or None
    """
    params = {
        "q": qword,
        "sort": "stars",
        "order": "desc",
        "per_page": 5
    }
    try:
        r = http_get(GITHUB_SEARCH_REPO, headers=GH_HEADERS, params=params, timeout=10)
        if not r:
            return None
        results = r.json()
        items = results.get("items", [])
        if items:
            return items[0].get("html_url")
    except Exception as e:
        logging.error(f"GitHub search error: {e}")
    return None

def find_code_repo(paper_title: str, arxiv_id_no_ver: str, primary_author: str | None = None) -> str | None:
    """
    Smarter GitHub fallback:
    1) Search for title phrase in README/description
    2) Search for arXiv ID
    3) Use Code Search to find arXiv ID in README files
    """
    try:
        # 1) Title phrase search
        q1 = f"\"{paper_title}\" in:readme,in:description"
        r = http_get(GITHUB_SEARCH_REPO, headers=GH_HEADERS,
                     params={"q": q1, "sort": "stars", "order": "desc", "per_page": 5}, timeout=10)
        if r and r.json().get("items"):
            return r.json()["items"][0]["html_url"]

        # 2) arXiv ID search
        q2 = f"\"{arxiv_id_no_ver}\" in:name,readme,description"
        r = http_get(GITHUB_SEARCH_REPO, headers=GH_HEADERS,
                     params={"q": q2, "sort": "stars", "order": "desc", "per_page": 5}, timeout=10)
        if r and r.json().get("items"):
            return r.json()["items"][0]["html_url"]

        # 3) Code Search: arXiv ID in README
        q3 = f"\"{arxiv_id_no_ver}\" in:file filename:README"
        r = http_get(GITHUB_SEARCH_CODE, headers=GH_HEADERS,
                     params={"q": q3, "per_page": 5}, timeout=10)
        if r and r.json().get("items"):
            return r.json()["items"][0]["repository"]["html_url"]
    except Exception as e:
        logging.error(f"find_code_repo error: {e}")
    return None

def get_repo_from_hf(arxiv_id_no_ver: str) -> str | None:
    """
    Get associated spaces/models/datasets from Hugging Face Hub.
    Priority: Spaces -> Models -> Datasets
    Returns the Hub link, or None on failure.
    """
    url = HF_REPOS_API.format(arxiv_id=arxiv_id_no_ver)
    try:
        r = http_get(url, headers=HF_HEADERS, timeout=10)
        if not r:
            return None
        data = r.json()  # {"models":[...], "datasets":[...], "spaces":[...]}

        def pick(arr, t):
            for it in (arr or []):
                rid = it.get("id")  # e.g., "org/name"
                if rid:
                    return f"https://huggingface.co/{t}/{rid}"
            return None

        return (pick(data.get("spaces"), "spaces")
                or pick(data.get("models"), "models")
                or pick(data.get("datasets"), "datasets"))
    except Exception as e:
        logging.error(f"HF repos error: {e}")
        return None

def _iter_arxiv_results(query: str, n: int):
    """
    Encapsulates arxiv.Client().results() and retries with a smaller max_results
    on UnexpectedEmptyPageError.
    """
    # Instantiate the client once
    client = arxiv.Client(
        page_size = 100,
        delay_seconds = 3,
        num_retries = 3
    )
    
    # Create the search object
    search = arxiv.Search(
        query=query, 
        max_results=n, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    try:
        # Use the recommended client.results(search) method
        for r in client.results(search):
            yield r
    except arxiv.UnexpectedEmptyPageError:
        logging.warning("Empty page from arXiv; retrying with fewer results (<=25)")
        # Fallback search with a smaller max_results
        search_fallback = arxiv.Search(
            query=query, 
            max_results=min(n, 25), 
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        try:
            # Use the same client for the fallback
            for r in client.results(search_fallback):
                yield r
        except Exception as e:
            logging.error(f"Error during fallback arXiv search: {e}")
    except Exception as e:
        # Catch other potential search errors
        logging.error(f"Error during arXiv search: {e}")

def get_daily_papers(topic,query="slam", max_results=2):
    """
    @param topic: str
    @param query: str
    @return paper_with_code: dict
    """
    content = {}
    content_to_web = {}

    for result in _iter_arxiv_results(query, max_results):

        paper_id            = result.get_short_id()      # e.g., 2108.09112v1
        paper_title         = result.title
        paper_url           = result.entry_id
        paper_abstract      = (result.summary or "").replace("\n"," ")
        paper_authors       = get_authors(result.authors)
        paper_first_author  = get_authors(result.authors,first_author = True)
        primary_category    = result.primary_category
        publish_time        = result.published.date() if result.published else ""
        update_time         = result.updated.date() if result.updated else publish_time
        comments            = result.comment

        logging.info(f"Time = {update_time} title = {paper_title} author = {paper_first_author}")

        # Remove version number: 2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        paper_key = paper_id if ver_pos == -1 else paper_id[:ver_pos]
        paper_url = arxiv_url + 'abs/' + paper_key

        # Try HF first, then GitHub search as fallback
        repo_url = get_repo_from_hf(paper_key)
        if repo_url is None:
            repo_url = (find_code_repo(paper_title, paper_key, paper_first_author)
                        or get_code_link(paper_title)
                        or get_code_link(paper_key))

        try:
            if repo_url is not None:
                content[paper_key] = "|**{}**|**{}**|{} et.al.|[{}]({})|**[link]({})**|\n".format(
                        update_time,paper_title,paper_first_author,paper_key,paper_url,repo_url)
                content_to_web[paper_key] = "- {}, **{}**, {} et.al., Paper: [{}]({}), Code: **[{}]({})**".format(
                        update_time,paper_title,paper_first_author,paper_url,paper_url,repo_url,repo_url)
            else:
                content[paper_key] = "|**{}**|**{}**|{} et.al.|[{}]({})|null|\n".format(
                        update_time,paper_title,paper_first_author,paper_key,paper_url)
                content_to_web[paper_key] = "- {}, **{}**, {} et.al., Paper: [{}]({})".format(
                        update_time,paper_title,paper_first_author,paper_url,paper_url)

            comments = None  # TODO: Retain comment logic if needed
            if comments != None:
                content_to_web[paper_key] += f", {comments}\n"
            else:
                content_to_web[paper_key] += f"\n"

        except Exception as e:
            logging.error(f"exception: {e} with id: {paper_key}")

    data = {topic:content}
    data_web = {topic:content_to_web}
    return data,data_web

def update_paper_links(filename):
    '''
    weekly update paper links in json file
    '''
    def parse_arxiv_string(s):
        parts = s.split("|")
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        pdf_field = parts[4].strip() # This is '[key](url)'
        code = parts[5].strip()
        # Extract the key from the markdown link
        arxiv_id_match = re.search(r'\[(.*?)\]', pdf_field)
        arxiv_id = arxiv_id_match.group(1).strip() if arxiv_id_match else pdf_field
        arxiv_id = re.sub(r'v\d+', '', arxiv_id) # Remove version
        return date, title, authors, arxiv_id, code, pdf_field # Pass back the original pdf_field

    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

        json_data = m.copy()

        for keywords,v in json_data.items():
            logging.info(f'keywords = {keywords}')
            # Use list(v.items()) to create a copy for safe iteration while modifying the dict
            for paper_id,contents in list(v.items()):
                contents = str(contents)
                try:
                    update_time, paper_title, paper_first_author, paper_key, code_url, paper_url_field = parse_arxiv_string(contents)

                    # Maintain original format
                    contents = "|{}|{}|{}|{}|{}|\n".format(update_time,paper_title,paper_first_author,paper_url_field,code_url)
                    json_data[keywords][paper_id] = str(contents)
                    # logging.info(f'paper_id = {paper_id}, contents = {contents}') # Too verbose

                    valid_link = False if '|null|' in contents else True
                    if valid_link:
                        continue
                    
                    # If link is null, try to find it again
                    logging.info(f"Attempting to find link for paper: {paper_key}")
                    repo_url = (get_repo_from_hf(paper_key)
                                or find_code_repo(paper_title, paper_key, paper_first_author)
                                or get_code_link(paper_title)
                                or get_code_link(paper_key))

                    if repo_url is not None:
                        new_cont = contents.replace('|null|',f'|**[link]({repo_url})**|')
                        logging.info(f'ID = {paper_key}, FOUND LINK, new contents = {new_cont}')
                        json_data[keywords][paper_id] = str(new_cont)

                except Exception as e:
                    logging.error(f"Exception processing paper_id: {paper_id}, contents: {contents}, error: {e}")
                    
        # dump to json file
        with open(filename,"w") as f:
            json.dump(json_data,f)

def update_json_file(filename,data_dict):
    '''
    daily update json file using data_dict
    '''
    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    # update papers in each keywords
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]

            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename,"w") as f:
        json.dump(json_data,f)

def json_to_md(filename,md_filename,
               task = '',
               to_web = False,
               use_title = True,
               use_tc = True,
               show_badge = True,
               use_b2t = True):
    """
    @param filename: str
    @param md_filename: str
    @return None
    """
    def pretty_math(s:str) -> str:
        ret = ''
        match = re.search(r"\$.*\$", s)
        if match == None:
            return s
        math_start,math_end = match.span()
        space_trail = space_leading = ''
        # Add boundary checks
        if math_start > 0 and s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]: space_trail = ' '
        if math_end < len(s) and s[math_end:][0] != ' ' and '*' != s[math_end:][0]: space_leading = ' '
        ret += s[:math_start]
        ret += f'{space_trail}${match.group()[1:-1].strip()}${space_leading}'
        ret += s[math_end:]
        return ret

    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-','.')

    with open(filename,"r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    # clean README.md if daily already exist else create it
    with open(md_filename,"w+") as f:
        pass

    # write data into README.md
    with open(md_filename,"a+") as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")

        if show_badge == True:
            f.write(f"[![Contributors][contributors-shield]][contributors-url]\n")
            f.write(f"[![Forks][forks-shield]][forks-url]\n")
            f.write(f"[![Stargazers][stars-shield]][stars-url]\n")
            f.write(f"[![Issues][issues-shield]][issues-url]\n\n")

        if use_title == True:
            f.write("## Updated on " + DateNow + "\n")
        else:
            f.write("> Updated on " + DateNow + "\n")

        f.write("> Usage instructions: [here](./docs/README.md#usage)\n\n")

        #Add: table of contents
        if use_tc == True:
            f.write("<details>\n")
            f.write("  <summary>Table of Contents</summary>\n")
            f.write("  <ol>\n")
            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                kw = keyword.replace(' ','-')
                f.write(f"    <li><a href=#{kw.lower()}>{keyword}</a></li>\n")
            f.write("  </ol>\n")
            f.write("</details>\n\n")

        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            # the head of each part
            f.write(f"## {keyword}\n\n")

            if use_title == True :
                if to_web == False:
                    f.write("|Publish Date|Title|Authors|PDF|Code|\n" + "|---|---|---|---|---|\n")
                else:
                    f.write("| Publish Date | Title | Authors | PDF | Code |\n")
                    f.write("|:---------|:-----------------------|:---------|:------|:------|\n")

            # sort papers by date
            day_content = sort_papers(day_content)

            for _,v in day_content.items():
                if v is not None:
                    f.write(pretty_math(v)) # make latex pretty

            f.write(f"\n")

            #Add: back to top
            if use_b2t:
                top_info = f"#Updated on {DateNow}"
                top_info = top_info.replace(' ','-').replace('.','')
                f.write(f"<p align=right>(<a href={top_info.lower()}>back to top</a>)</p>\n\n")

        if show_badge == True:
            # we don't like long string, break it!
            f.write((f"[contributors-shield]: https://img.shields.io/github/"
                     f"contributors/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[contributors-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/graphs/contributors\n"))
            f.write((f"[forks-shield]: https://img.shields.io/github/forks/Vincentqyw/"
                     f"cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[forks-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/network/members\n"))
            f.write((f"[stars-shield]: https://img.shields.io/github/stars/Vincentqyw/"
                     f"cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[stars-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/stargazers\n"))
            f.write((f"[issues-shield]: https://img.shields.io/github/issues/Vincentqyw/"
                     f"cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[issues-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/issues\n\n"))

    logging.info(f"{task} finished")

def demo(**config):
    data_collector = []
    data_collector_web= []

    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    publish_wechat = config['publish_wechat']
    show_badge = config['show_badge']

    b_update = config['update_paper_links']
    logging.info(f'Update Paper Link = {b_update}')
    if config['update_paper_links'] == False:
        logging.info(f"GET daily papers begin")
        for topic, keyword in keywords.items():
            logging.info(f"Keyword: {topic}")
            data, data_web = get_daily_papers(topic, query = keyword,
                                            max_results = max_results)
            data_collector.append(data)
            data_collector_web.append(data_web)
            print("\n")
        logging.info(f"GET daily papers end")

    # 1. update README.md file
    if publish_readme:
        json_file = config['json_readme_path']
        md_file   = config['md_readme_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file,data_collector)
        json_to_md(json_file,md_file, task ='Update Readme', show_badge = show_badge)

    # 2. update docs/index.md file (to gitpage)
    if publish_gitpage:
        json_file = config['json_gitpage_path']
        md_file   = config['md_gitpage_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file,data_collector)
        json_to_md(json_file, md_file, task ='Update GitPage',
                   to_web = True, show_badge = show_badge,
                   use_tc=False, use_b2t=False)

    # 3. Update docs/wechat.md file
    if publish_wechat:
        json_file = config['json_wechat_path']
        md_file   = config['md_wechat_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector_web)
        json_to_md(json_file, md_file, task ='Update Wechat',
                   to_web=False, use_title= False, show_badge = show_badge)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml',
                            help='configuration file path')
    parser.add_argument('--update_paper_links', default=False,
                            action="store_true",help='whether to update paper links etc.')
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_paper_links':args.update_paper_links}
    demo(**config)

    # The git commit and push logic is handled by the GitHub Action workflow
    # (e.g., using github-actions-x/commit).
    # Removing redundant subprocess calls that were here.
    logging.info("Script finished. Git commit and push will be handled by the workflow.")
