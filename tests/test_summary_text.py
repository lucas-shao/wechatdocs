from wechatdocs.summary.summary_text import summary_text

with open("resource/contract.txt") as f:
    contract = f.read()

summary_text(contract, 200)
