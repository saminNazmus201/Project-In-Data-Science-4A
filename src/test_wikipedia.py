import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent="ProportionalRAGResearch/1.0 (academic project)",
    language="en",
)

page = wiki.page("Christine Sinclair")

print("Page exists:", page.exists())
print("Title:", page.title)
print("Text preview:")
print(page.text[:1000])