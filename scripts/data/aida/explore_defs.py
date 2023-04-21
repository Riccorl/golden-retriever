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


definitions_dpr_sets = set()
definitions_titles_sets = set()

for data_set in ["train", "val", "test"]:
    with open(
        f"/root/golden-retriever/data/aida_dpr_special_tok/{data_set}.json", "r"
    ) as f:
        data = json.load(f)
        for line in data:
            # line_data = json.loads(line)
            for definition in line["positive_ctxs"]:
                definitions_dpr_sets.add(definition["text"])
                definitions_titles_sets.add(definition["title"])

# check if they are present in the whole index
total_titles = set()
total_dpr = set()
with open("/root/golden-retriever/data/aida_dpr_special_tok/definitions.txt") as f:
    for line in f:
        total_dpr.add(line[:-1])
        total_titles.add(line.split(" <def> ")[0])

print(len(total_dpr))
print(len(total_titles))

print(
    "intersection definitions_titles and total_titles",
    len(definitions_titles_sets.intersection(total_titles)),
)
print(
    "intersection definitions_dpr_sets and total_dpr",
    len(definitions_dpr_sets.intersection(total_dpr)),
)

print(definitions_titles_sets.difference(total_titles))
# print missing definitions in dpr
