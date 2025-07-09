import codecs
import os

def fix_template_encoding():
    template_path = os.path.join('app', 'templates', 'index.html')
    # Read with utf-8-sig to handle BOM
    with codecs.open(template_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    # Write without BOM
    with codecs.open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    fix_template_encoding()
    print("Template encoding fixed successfully.") 