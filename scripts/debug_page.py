with open('generated-page.html', 'r', encoding='utf-8') as f:
    text = f.read()

import re
events = re.findall(r"addEventListener\(['\"](\w+)['\"]", text)
print("Event listeners in generated-page.html:", set(events))

for m in re.finditer(r"heroGL", text):
    idx = m.start()
    print("--- heroGL at", idx)
    print(text[max(0, idx-100):min(len(text), idx+150)])

for m in re.finditer(r"pointermove", text):
    idx = m.start()
    print("--- pointermove at", idx)
    print(text[max(0, idx-100):min(len(text), idx+150)])
