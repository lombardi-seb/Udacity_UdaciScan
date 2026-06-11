from config import load_settings
settings = load_settings()

import sys
sys.path.append("/workspace/code") # set this to the path containing the "ls_action_space" package

from ls_action_space.action_space import query_pubmed, extract_pdf_content

if __name__ == "__main__":
    if settings.email == "student@udacity.com":
        print("Please make sure you enter your email address in config.yaml")
        print("(needed as your identifier for the NCBI Entrez and unpaywall.org APIs - you will not be spammed)")
    else:
        test_query = "Parkinson's"
        print("Searching for articles with query: ",test_query)
        articles = query_pubmed(test_query)
        for a in articles:
            print(a)
            doi, pdf_text = "UNKNOWN", ""
            try:
                doi = a.get("doi")
                pdf_text = extract_pdf_content(doi)
            except Exception:
                print("Unable to locate PDF for DOI:",doi)
            print(pdf_text)
            print()
        print("Done")