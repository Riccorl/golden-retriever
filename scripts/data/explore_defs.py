import json

# definitions_kilt = set()
# with open("/root/golden-retriever/data/aida/entities_kilt.json", "r") as f:
#     for line in f:
#         line_data = json.loads(line)
#         definitions_kilt.add(line_data["title"])

# definitions_dpr = set()
# with open("/root/golden-retriever/data/aida/definitions.txt", "w") as f_write:
#     with open("/root/golden-retriever/data/aida/entity.jsonl", "r") as f:
#         for line in f:
#             line_data = json.loads(line)
#             definitions_dpr.add(line_data["title"])
#             f_write.write(line_data["title"] + ": " + line_data["text"] + "\n")

# print(len(definitions_kilt))
# print(len(definitions_dpr))
# print(len(definitions_kilt.intersection(definitions_dpr)))
# # print missing definitions in dpr
# print(definitions_kilt.difference(definitions_dpr))

definitions_dpr = set()
for set in ['train', 'val', 'test']:
    with open(f"/root/golden-retriever/data/aida_dpr/{set}.json", "r") as f:
        data = json.load(f)
        for line in data:
            # line_data = json.loads(line)
            for definition in line["positive_ctxs"]:
                definitions_dpr.add(definition["text"])

with open("/root/golden-retriever/data/aida/definitions_aida.txt", "w") as f_write:
    for definition in definitions_dpr:
        f_write.write(definition + "\n")
