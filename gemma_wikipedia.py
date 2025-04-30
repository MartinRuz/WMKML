import os.path

import wikipediaapi
import re
import json

def extract_sections(sections, level=0):
    extracted = []
    for section in sections:
        if section.title not in ['Religion', 'Bereits feststehende Ereignisse', 'Voraussichtliche Ereignisse',
                                 'Wissenschaftspreise', 'Jahrestage', 'Gedenktage', 'Kulturelle Referenzen',
                                 'Gestorben', 'Galerie der Verstorbenen', 'Weblinks', 'Einzelnachweise', 'Sport',
                                 'Wissenschaft und Technik', 'Jahreswidmungen']:
            extracted.append({
                'title': section.title,
                'text': section.text.strip(),
                'level': level
            })
            extracted.extend(extract_sections(section.sections, level+1))
    return extracted


def extract_date_event_pairs(text):
    #pattern = r"(?m)^\s*[-–•*]?\s*(\d{1,2}\.\s*[A-ZÄÖÜa-zäöü]+):\s*(.*)"
    pattern = r"*(\r\n|\r|\n)"
    matches = re.findall(pattern, text)
    return [{"date": date.strip(), "event": event.strip()} for date, event in matches]


if __name__ == "__main__":
    if os.path.exists('latest_events.json'):
        with open('latest_events.json', 'r', encoding='utf-8') as f:
            events = json.load(f)
        qa_dataset = []
        for event in events:
            qa_dataset.append({
                "question": f"Was geschah am {event['date']}?",
                "answer": event['event']
            })
        with open("latest_qa_dataset.jsonl", "w", encoding="utf-8") as f:
            for dataset in qa_dataset:
                f.write(json.dumps(dataset, ensure_ascii=False) + "\n")

        rag_docs = []
        for idx, item in enumerate(events):
            rag_docs.append({
                "id": f"{item['date'].replace(' ', '_')}_{idx}",
                "text": f"{item['date']}: {item['event']}",
            })

        with open("latest_rag_docs.jsonl", "w", encoding="utf-8") as f:
            for doc in rag_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    else:
        wiki = wikipediaapi.Wikipedia(user_agent='Extraction of latest world events (martin.ruzicka@gmx.de)', language='de')
        page = wiki.page('2024')
        page2 = wiki.page('2025')

        sections = extract_sections(page.sections)
        sections2 = extract_sections(page2.sections)
        all_sections = sections + sections2
        event_pairs = []

        for section in all_sections:
            text = section['text']
            lines = text.split("\n")
            for line in lines:
                date = line.split(":")
                if len(date) == 2:
                    data = {"date": date[0], "event": date[1]}
                    event_pairs.append(data)

        with open('latest_events.json', 'w', encoding='utf-8') as outfile:
            json.dump(event_pairs, outfile, ensure_ascii=False, indent=2)

        print(f"{len(event_pairs)} events extracted")
        for event in event_pairs[:5]:
            print(f"{event['date']} {event['event']}")
