import io
import re
from datetime import date, timedelta
import time
from typing import List, Optional, Tuple
import pandas as pd
import streamlit as st
from Bio import Entrez


# ============================================================
# App setup
# ============================================================

st.set_page_config(
    page_title="PubMed Search Tool",
    page_icon="🔍",
    layout="wide",
)

st.title("PubMed Search Tool")
st.caption(
    "Search PubMed by author names, then optionally filter by topic, affiliation, and publication date."
)


# ============================================================
# Helpers: text parsing / normalization
# ============================================================

def clean_lines(text: str) -> List[str]:
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_csv_names(uploaded_file) -> List[str]:
    if uploaded_file is None:
        return []

    df = pd.read_csv(uploaded_file)
    if df.empty:
        return []

    name_col = None
    for candidate in ["name", "Name", "author", "Author", "person", "Person"]:
        if candidate in df.columns:
            name_col = candidate
            break

    if name_col is None:
        name_col = df.columns[0]

    return [str(x).strip() for x in df[name_col].dropna().tolist() if str(x).strip()]


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_name(text).lower()).strip()


def split_target_name(name: str) -> Tuple[str, str, str, bool]:
    """
    Returns:
        given_names, surname, initials, looks_like_pubmed_style

    Supports:
      - Bruce Wang
      - Wang B
      - Wang BM
    """
    cleaned = normalize_name(name)
    parts = cleaned.split()
    if not parts:
        return "", "", "", False

    # Already PubMed-ish input like "Wang B" or "Wang BM"
    if len(parts) >= 2 and len(parts[0]) > 1 and len(parts[-1]) <= 3 and parts[-1].replace(".", "").isalpha():
        surname = parts[0]
        initials = parts[1].replace(".", "")
        return "", surname, initials, True

    if len(parts) == 1:
        return "", parts[0], "", False

    surname = parts[-1]
    given = " ".join(parts[:-1])
    initials = "".join(p[0] for p in parts[:-1] if p and p[0].isalpha())
    return given, surname, initials, False


# ============================================================
# PubMed query builders
# ============================================================

def build_author_query(name: str) -> str:
    """
    Build a PubMed author query using PubMed-style indexing.

    Examples:
      Bruce Wang -> Wang B[AUTH]
      Wang B -> Wang B[AUTH]
    """
    given, surname, initials, already_indexed = split_target_name(name)
    if not surname:
        return ""

    clauses = []

    # Primary PubMed-style author query
    if initials:
        clauses.append(f'"{surname} {initials}"[AUTH]')
        clauses.append(f'"{surname} {initials}"[AU]')
        clauses.append(f'"{surname}"[AUTH]')
        clauses.append(f'"{surname}"[AU]')
    else:
        clauses.append(f'"{surname}"[AUTH]')
        clauses.append(f'"{surname}"[AU]')

    # If the user typed a full name, include it as a fallback
    if given:
        clauses.append(f'"{given} {surname}"[AUTH]')
        clauses.append(f'"{given} {surname}"[AU]')

    # Deduplicate while keeping order
    clauses = list(dict.fromkeys(clauses))
    return "(" + " OR ".join(clauses) + ")"


def build_topic_clause(topics: List[str]) -> str:
    """
    Match topic keywords in title/abstract/MeSH.
    Uses OR semantics across the list of topics.
    """
    terms = [t for t in topics if t]
    if not terms:
        return ""

    clauses = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        clauses.append(f'("{term}"[TIAB] OR "{term}"[MESH])')

    return "(" + " OR ".join(clauses) + ")" if clauses else ""


def build_affiliation_clause(affiliations: List[str]) -> str:
    """
    Match affiliation keywords in the PubMed AFFL field.
    Uses OR semantics across the list of affiliations.
    """
    terms = [a for a in affiliations if a]
    if not terms:
        return ""

    clauses = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        clauses.append(f'"{term}"[AFFL]')

    return "(" + " OR ".join(clauses) + ")" if clauses else ""


def build_query(name: str, topics: List[str], affiliations: List[str]) -> str:
    parts = [build_author_query(name)]

    topic_clause = build_topic_clause(topics)
    if topic_clause:
        parts.append(topic_clause)

    affl_clause = build_affiliation_clause(affiliations)
    if affl_clause:
        parts.append(affl_clause)

    parts = [p for p in parts if p]
    return " AND ".join(parts) if parts else ""


# ============================================================
# PubMed date presets
# ============================================================

def subtract_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year - years)


def get_date_params(date_mode: str, custom_start: Optional[date], custom_end: Optional[date]):
    """
    Return kwargs for Entrez.esearch date filtering.
    Uses pdat (publication date) per Biopython docs.
    """
    today = date.today()

    if date_mode == "All time":
        return {}
    elif date_mode == "Within 1 year":
        return {"datetype": "pdat", "reldate": 365}
    elif date_mode == "Within 5 years":
        return {
            "datetype": "pdat",
            "mindate": subtract_years(today, 5).strftime("%Y/%m/%d"),
            "maxdate": today.strftime("%Y/%m/%d"),
        }
    elif date_mode == "Within 10 years":
        return {
            "datetype": "pdat",
            "mindate": subtract_years(today, 10).strftime("%Y/%m/%d"),
            "maxdate": today.strftime("%Y/%m/%d"),
        }
    else:
        start = custom_start or subtract_years(today, 10)
        end = custom_end or today
        if start > end:
            start, end = end, start
        return {
            "datetype": "pdat",
            "mindate": start.strftime("%Y/%m/%d"),
            "maxdate": end.strftime("%Y/%m/%d"),
        }


# ============================================================
# PubMed XML helpers
# ============================================================

def parse_pub_date(pub_date: dict) -> str:
    year = pub_date.get("Year")
    if not year:
        return "Not Available"
    month = pub_date.get("Month", "01")
    day = pub_date.get("Day", "01")
    return f"{year}-{month}-{day}"


def safe_join_abstract(article: dict) -> str:
    abstract = article.get("Abstract")
    if not abstract:
        return ""
    return " ".join(str(x) for x in abstract.get("AbstractText", []))


def author_display_name(author: dict) -> str:
    collective = normalize_name(str(author.get("CollectiveName", "")))
    if collective:
        return collective

    last = normalize_name(str(author.get("LastName", "")))
    fore = normalize_name(str(author.get("ForeName", "")))
    if fore and last:
        return f"{fore} {last}".strip()
    return last or fore


def author_pubmed_index(author: dict) -> str:
    """
    Convert a PubMed author record to surname + initials.
    Example: Bruce Wang -> Wang B
    """
    last = normalize_name(str(author.get("LastName", "")))
    initials = normalize_name(str(author.get("Initials", ""))).replace(" ", "")
    if not initials:
        fore = normalize_name(str(author.get("ForeName", "")))
        initials = "".join(part[0] for part in fore.split() if part and part[0].isalpha())

    if last and initials:
        return normalize_for_match(f"{last} {initials}")
    if last:
        return normalize_for_match(last)
    return ""


def get_affiliations_from_article(article: dict) -> List[str]:
    affiliations = []
    for author in article.get("AuthorList", []):
        for aff in author.get("AffiliationInfo", []):
            aff_text = normalize_name(str(aff.get("Affiliation", "")))
            if aff_text:
                affiliations.append(aff_text)
    return list(dict.fromkeys(affiliations))


def get_mesh_terms(citation: dict) -> List[str]:
    mesh = []
    for item in citation.get("MeshHeadingList", []):
        descriptor = item.get("DescriptorName")
        if descriptor:
            mesh.append(str(descriptor))
    return list(dict.fromkeys(mesh))


def get_keywords(article: dict) -> List[str]:
    kws = []
    for kw in article.get("KeywordList", []):
        for item in kw:
            text = str(item).strip()
            if text:
                kws.append(text)
    return list(dict.fromkeys(kws))


def text_matches_any(haystack: str, terms: List[str]) -> Tuple[bool, List[str]]:
    if not terms:
        return True, []
    hay = normalize_for_match(haystack)
    matched = [t for t in terms if normalize_for_match(t) in hay]
    return (len(matched) > 0, matched)


def author_matches(author_list: List[dict], target_name: str) -> Tuple[bool, str]:
    """
    Exact match on first + last name from the PubMed author record.
    Example: Bruce Wang matches only Bruce Wang.
    """
    target_clean = normalize_name(target_name)
    target_norm = normalize_for_match(target_clean)

    for author in author_list:
        display = author_display_name(author)
        display_norm = normalize_for_match(display)

        # Exact first + last name match
        if display_norm == target_norm:
            return True, display

    return False, ""


# ============================================================
# Search via ESearch history, then EFetch in batches
# ============================================================

def fetch_pubmed_results_for_name(
    name: str,
    topics: List[str],
    affiliations: List[str],
    date_mode: str,
    custom_start: Optional[date],
    custom_end: Optional[date],
    batch_size: int,
    email: str,
) -> pd.DataFrame:
    Entrez.email = email

    query = build_query(name, topics, affiliations)
    if not query:
        return pd.DataFrame()

    search_kwargs = {
        "db": "pubmed",
        "term": query,
        "usehistory": "y",
        "sort": "pub date",
        "retmax": 0,
    }
    search_kwargs.update(get_date_params(date_mode, custom_start, custom_end))

    search_handle = Entrez.esearch(**search_kwargs)
    search_record = Entrez.read(search_handle)
    search_handle.close()

    count = int(search_record["Count"])
    if count == 0:
        return pd.DataFrame()

    webenv = search_record["WebEnv"]
    query_key = search_record["QueryKey"]

    rows = []
    seen_pmids = set()

    for start in range(0, count, batch_size):
        fetch_handle = Entrez.efetch(
            db="pubmed",
            rettype="xml",
            retmode="xml",
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key,
        )
        fetched = Entrez.read(fetch_handle)
        fetch_handle.close()

        for record in fetched.get("PubmedArticle", []):
            citation = record.get("MedlineCitation", {})
            article = citation.get("Article", {})
            pmid = str(citation.get("PMID", "")).strip()

            if not pmid or pmid in seen_pmids:
                continue

            author_list = article.get("AuthorList", [])
            author_ok, matched_author = author_matches(author_list, name)
            if not author_ok:
                continue

            title = str(article.get("ArticleTitle", "Title Not Available"))
            abstract = safe_join_abstract(article)
            affiliations_found = get_affiliations_from_article(article)
            mesh_terms = get_mesh_terms(citation)
            keywords = get_keywords(article)

            journal = article.get("Journal", {}).get("Title", "Journal Not Available")
            pub_date = parse_pub_date(article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}))

            # title / abstract / MeSH / keywords are all good places to match topic terms
            topic_text = " | ".join([title, abstract, "; ".join(mesh_terms), "; ".join(keywords)])
            topic_ok, matched_topics = text_matches_any(topic_text, topics)

            affl_text = " | ".join(affiliations_found)
            affl_ok, matched_affils = text_matches_any(affl_text, affiliations)

            if not topic_ok:
                continue
            if affiliations and not affl_ok:
                continue

            doi = ""
            pmcid = ""
            for aid in record.get("PubmedData", {}).get("ArticleIdList", []):
                aid_type = getattr(aid, "attributes", {}).get("IdType")
                if aid_type == "doi":
                    doi = str(aid)
                elif aid_type == "pmc":
                    pmcid = str(aid)

            authors_display = [author_display_name(a) for a in author_list if author_display_name(a)]
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            rows.append(
                {
                    "Query Name": name,
                    "Matched Author": matched_author,
                    "PMID": pmid,
                    "PMCID": pmcid,
                    "DOI": doi,
                    "Title": title,
                    "Authors": ", ".join(authors_display) if authors_display else "Authors Not Available",
                    "Matched Topics": ", ".join(matched_topics),
                    "Matched Affiliations": ", ".join(matched_affils),
                    "Affiliations": " | ".join(affiliations_found),
                    "Journal": journal,
                    "Publication Date": pub_date,
                    "PubMed URL": url,
                    "Abstract": abstract,
                    "MeSH Terms": ", ".join(mesh_terms),
                    "Keywords": ", ".join(keywords),
                }
            )
            seen_pmids.add(pmid)

        time.sleep(0.34)  # stay under NCBI’s default 3 req/sec guidance

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["PMID"])
        if "Publication Date" in df.columns:
            df = df.sort_values(by=["Publication Date", "PMID"], ascending=[False, True])
    return df


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PubMed Results")
        ws = writer.book["PubMed Results"]

        # Turn URL column into clickable hyperlinks
        if "PubMed URL" in df.columns:
            url_col = list(df.columns).index("PubMed URL") + 1
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=url_col)
                url = cell.value
                if isinstance(url, str) and url.startswith("http"):
                    cell.hyperlink = url
                    cell.style = "Hyperlink"
                    cell.value = "PubMed link"

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # Basic width tuning
        widths = {
            "A": 18, "B": 22, "C": 12, "D": 12, "E": 20, "F": 55,
            "G": 35, "H": 25, "I": 28, "J": 55, "K": 35, "L": 16,
            "M": 16, "N": 55, "O": 40, "P": 40,
        }
        for col_letter, width in widths.items():
            ws.column_dimensions[col_letter].width = width

    output.seek(0)
    return output.getvalue()


# ============================================================
# UI
# ============================================================

with st.sidebar:
    st.header("Search settings")
    email = st.text_input("NCBI email", value="your.email@example.com")
    batch_size = st.number_input(
        "Fetch batch size",
        min_value=1,
        max_value=200,
        value=25,
        step=1,
        help="How many PubMed records to fetch at a time.",
    )
    date_mode = st.radio(
        "Publication date",
        ["All time", "Within 1 year", "Within 5 years", "Within 10 years", "Custom Range"],
        index=0,
    )

    custom_start = None
    custom_end = None
    if date_mode == "Custom Range":
        custom_start = st.date_input("Start date", value=subtract_years(date.today(), 10))
        custom_end = st.date_input("End date", value=date.today())

    require_affiliation = st.checkbox(
        "Require affiliation match",
        value=False,
        help="Leave off for broader results.",
    )

st.subheader("1) Add names")
name_text = st.text_area(
    "Paste one name per line",
    value="Bruce Wang\nWang B",
    height=120,
    help="You can also upload a CSV with a name column.",
)

uploaded = st.file_uploader(
    "Or upload a CSV with names",
    type=["csv"],
    accept_multiple_files=False,
)

manual_names = clean_lines(name_text)
csv_names = parse_csv_names(uploaded)
all_names = [normalize_name(n) for n in manual_names + csv_names]
all_names = [n for i, n in enumerate(all_names) if n and n not in all_names[:i]]

st.write(f"**Names loaded:** {len(all_names)}")
if all_names:
    st.write(", ".join(all_names))

st.subheader("2) Add topic, affiliation, and keyword filters")
col1, col2 = st.columns(2)

with col1:
    topic_text = st.text_area(
        "Topic keywords, one per line",
        value="liver\nporphyria",
        height=140,
        help="Matched against title, abstract, MeSH terms, and keywords.",
    )
    affiliation_text = st.text_area(
        "Affiliation keywords, one per line",
        value="UCSF\nUniversity of California San Francisco",
        height=140,
        help="Matched against PubMed AFFL plus affiliations found in the XML record.",
    )

with col2:
    st.info(
        "Examples:\n\n"
        "- Topic: liver\n"
        "- Topic: porphyria\n"
        "- Affiliation: UCSF\n"
        "- Affiliation: University of California San Francisco"
    )

topics = clean_lines(topic_text)
affiliations = clean_lines(affiliation_text)

st.subheader("3) Search and export")
search_button = st.button("Search PubMed", type="primary")

if search_button:
    if not all_names:
        st.error("Please enter at least one name or upload a CSV file with names.")
        st.stop()

    if not email or "@" not in email:
        st.error("Please enter a valid email address for NCBI.")
        st.stop()

    with st.spinner("Searching PubMed..."):
        all_results = []
        for name in all_names:
            try:
                df = fetch_pubmed_results_for_name(
                    name=name,
                    topics=topics,
                    affiliations=affiliations if require_affiliation else [],
                    date_mode=date_mode,
                    custom_start=custom_start,
                    custom_end=custom_end,
                    batch_size=int(batch_size),
                    email=email,
                )
                if not df.empty:
                    all_results.append(df)
            except Exception as e:
                st.error(f"Search failed for {name}: {e}")

        results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    if results.empty:
        st.warning("No results found.")
    else:
        # keep one row per PMID
        results = results.drop_duplicates(subset=["PMID"]).reset_index(drop=True)

        st.success(f"Found {len(results)} publications.")
        st.dataframe(results, use_container_width=True, hide_index=True)

        excel_bytes = dataframe_to_excel_bytes(results)
        st.download_button(
            label="Download Excel",
            data=excel_bytes,
            file_name="PubMed_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button(
            label="Download CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="PubMed_results.csv",
            mime="text/csv",
        )

st.divider()
st.markdown(
    """
**How it works**
- Enter a list of names or upload a CSV.
- PubMed author search uses `AUTH` / `AU`-style indexing.
- Topics are searched in title, abstract, MeSH, and keywords.
- Affiliations are searched with `AFFL`.
- Results are fetched through PubMed history (`WebEnv` / `QueryKey`) in batches.
"""
)