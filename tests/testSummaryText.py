from wechatdocs.summary.summaryText import summaryText

with open("resource/contract.txt") as f:
    contract = f.read()

summaryText(contract, 200)
