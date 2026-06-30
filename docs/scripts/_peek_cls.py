from pathlib import Path
import re

h = Path(r"D:\wsl\raha\docs\index.html").read_text(encoding="utf-8")
i = h.find('id="classification"')
section = h[i:h.find("</section>", i) + 10]
section = re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+', "[b64]", section)
section = re.sub(r'src="res/[^"]+"', 'src="res/..."', section)
print(section)
