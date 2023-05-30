import re

def clean_and_extract_spans(text):
    # find all text spans and their positions
    text_spans = [(match.start(), match.end()) for match in re.finditer(r'{\s.*?\s}', text)]
    # find all label spans and their positions
    label_spans = [(match.start(), match.end()) for match in re.finditer(r'\[\s.*?\s\]', text)]
    # create a dictionary to store the extracted spans
    spans_dict = {}
    # iterate over the text spans
    for span in text_spans:
        # find the corresponding label span
        for label_span in label_spans:
            # if the label span starts after the text span, break the loop
            if label_span[0] > span[1]:
                break
            # if the label span is inside the text span, extract the span and label
            if label_span[0] > span[0] and label_span[1] < span[1]:
                span_text = text[span[0]+2:span[1]-1]
                label_text = text[label_span[0]+2:label_span[1]-1]
                spans_dict[str(span)] = span_text
                break
    return spans_dict

text = "{ United Kingdoms } [ United Kingdoms ] may refer to: { United Kingdoms of Denmark–Norway } [ Denmark–Norway ]."
print(clean_and_extract_spans(text))
