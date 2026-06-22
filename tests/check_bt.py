with open('abuja.html', 'r', encoding='utf-8') as f:
    html = f.read()
print('Number of backticks:', html.count('`'))
