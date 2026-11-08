# Author: William Degrer
import argparse
import json
import os
import random

CANDIDATES_FILE = "data/space/processed/space_retrieved_reviews.json"
METHODS = ["Top-K", "MMR", "Proportional"]
QUESTIONS = [
    ["representativeness", "Representative: does it reflect what most of the reviews says, in the right proportions?"],
    ["informativeness", "Informative: does it tell you something useful about the hotel?"],
    ["trust", "Trust: would you trust this summary when you choose hotel?"],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--n_cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", default="results/user_study")
    args = parser.parse_args()

    with open(args.summaries, encoding="utf-8") as f:
        items = json.load(f)["data"]
    by_case = {}
    for it in items:
        key = (it["hotel_id"], it["aspect"])
        if key not in by_case:
            by_case[key] = {}
        by_case[key][it["method"]] = it
    pools = {}
    with open(CANDIDATES_FILE) as f:
        for c in json.load(f):
            pools[(str(c["hotel_id"]), c["aspect"])] = c["reviews"]

    rng = random.Random(args.seed)
    complete = []
    for k in sorted(by_case):
        ok = True
        for m in METHODS:
            if m not in by_case[k] or not by_case[k][m].get("output"):
                ok = False
        if ok:
            complete.append(k)

    # half cleanliness half service, never the same hotel twice
    chosen = []
    for aspect in ["cleanliness", "service"]:
        used = [h for h, a in chosen]
        options = [k for k in complete if k[1] == aspect and k[0] not in used]
        chosen += rng.sample(options, args.n_cases // 2)
    rng.shuffle(chosen)

    key = []
    cases = []
    for n, (hotel, aspect) in enumerate(chosen):
        order = list(METHODS)
        rng.shuffle(order)
        letters = {"A": order[0], "B": order[1], "C": order[2]}
        key.append({"case": n + 1, "hotel_id": hotel, "aspect": aspect, "letters": letters})
        pool = pools[(hotel, aspect)]
        mix = {"positive": 0, "neutral": 0, "negative": 0}
        for r in pool:
            mix[r["sentiment"]] += 1
        summaries = []
        for L in ["A", "B", "C"]:
            summaries.append({"letter": L, "text": by_case[(hotel, aspect)][letters[L]]["output"]})
        cases.append({
            "case": n + 1,
            "question": by_case[(hotel, aspect)][METHODS[0]]["question"],
            "mix": mix,
            "pool": [{"stars": int(round(float(r["rating"]))), "text": r["text"]} for r in pool],
            "summaries": summaries,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_dir + "/key.json", "w") as f:
        json.dump({"seed": args.seed, "source": args.summaries, "cases": key}, f, indent=2)
    with open(args.out_dir + "/user_study.html", "w", encoding="utf-8") as f:
        f.write(render(cases))
    print("Wrote", args.out_dir + "/user_study.html", "and key.json,", len(cases), "cases")


def render(cases):
    data = json.dumps({"cases": cases, "questions": QUESTIONS}, ensure_ascii=False)
    data = data.replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hotel Summary Study</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a65;--line:#e2e0d8;--accent:#2f5d50;--pos:#3b7d4f;--neu:#a68a2c;--neg:#b0463c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#171816;--card:#21231f;--ink:#ecebe6;--muted:#a3a29b;--line:#363832;--accent:#7fb8a6;--pos:#7fbf8f;--neu:#d6b85c;--neg:#e0857b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 80px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.15rem;margin:0 0 6px}
.lead{color:var(--muted);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:0 0 18px}
.mix{display:flex;height:12px;border-radius:6px;overflow:hidden;margin:8px 0 4px}.mix span{display:block}
.legend{font-size:.85rem;color:var(--muted)}details{margin:8px 0 14px}summary{cursor:pointer;color:var(--accent)}
.rev{font-size:.88rem;border-top:1px solid var(--line);padding:6px 0}.stars{font-weight:600;margin-right:6px}
.sum{border-left:3px solid var(--accent);padding:6px 12px;margin:14px 0 8px;white-space:pre-wrap}
.q{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;font-size:.92rem;margin:4px 0}
select,input{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
button{font:inherit;background:var(--accent);color:var(--card);border:0;border-radius:8px;padding:10px 18px;cursor:pointer}
.err{color:var(--neg)}@media (max-width:560px){.q{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Hotel review summaries</h1>
<p class="lead">For each hotel you see a question, the opinion mix of the 20 most relevant guest reviews and three short summaries (A, B, C) that is written by an AI from different subsets of the reviews. Give every summary a score from 1 (very bad) to 10 (excellent). There is no right answers. It takes around 20 minutes.</p>
<div class="card"><label>Your name or initials: <input id="who" autocomplete="off"></label></div>
<div id="cases"></div>
<div class="card"><button id="dl">Download my answers (CSV)</button> <span id="msg"></span>
<p class="legend">Then send the downloaded file back to the person who gave you this form, thanks for helping!</p></div>
</main>
<script>
const DATA = __DATA__;
const root = document.getElementById('cases');
const colors = {positive:'var(--pos)', neutral:'var(--neu)', negative:'var(--neg)'};
function el(tag, attrs, ...kids){const e=document.createElement(tag);Object.assign(e,attrs||{});kids.forEach(k=>e.append(k));return e;}
DATA.cases.forEach(c=>{
  const card = el('section',{className:'card'});
  card.append(el('h2',{textContent:`Hotel ${c.case} of ${DATA.cases.length}: ${c.question}`}));
  const total = c.mix.positive + c.mix.neutral + c.mix.negative;
  const bar = el('div',{className:'mix'});
  ['positive','neutral','negative'].forEach(s=>{const sp=el('span');sp.style.width=(100*c.mix[s]/total)+'%';sp.style.background=colors[s];bar.append(sp);});
  card.append(bar, el('div',{className:'legend',textContent:`Of the ${total} most relevant reviews: ${c.mix.positive} positive (4-5 stars), ${c.mix.neutral} neutral (3), ${c.mix.negative} negative (1-2).`}));
  const det = el('details'); det.append(el('summary',{textContent:'Read the reviews'}));
  c.pool.forEach(r=>{const d=el('div',{className:'rev'});d.append(el('span',{className:'stars',textContent:'★'.repeat(r.stars)}), r.text);det.append(d);});
  card.append(det);
  c.summaries.forEach(s=>{
    card.append(el('div',{className:'sum',textContent:`Summary ${s.letter}\n${s.text}`}));
    DATA.questions.forEach(([key,label])=>{
      const sel = el('select',{name:`${c.case}|${s.letter}|${key}`});
      sel.append(el('option',{value:'',textContent:'–'}));
      for(let i=1;i<=10;i++) sel.append(el('option',{value:i,textContent:i}));
      const row = el('label',{className:'q'}); row.append(el('span',{textContent:label}), sel); card.append(row);
    });
  });
  root.append(card);
});
document.getElementById('dl').onclick = ()=>{
  const who = document.getElementById('who').value.trim() || 'anonymous';
  const sels = [...document.querySelectorAll('select')];
  const missing = sels.filter(s=>!s.value).length;
  const msg = document.getElementById('msg');
  msg.className = missing ? 'err' : ''; msg.textContent = missing ? `${missing} scores is missing, you can still download.` : 'All scores filled in.';
  const lines = ['evaluator,case,letter,criterion,score'];
  sels.forEach(s=>{const [c,l,q]=s.name.split('|'); lines.push([JSON.stringify(who),c,l,q,s.value].join(','));});
  const a = el('a',{href:URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'})),download:`user_study_${who.replace(/\W+/g,'_')}.csv`});
  document.body.append(a); a.click(); a.remove();
};
</script></body></html>
"""

if __name__ == "__main__":
    main()
