import editdistance

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if target_text == "":
        if predicted_text == "":
            return 0
        return 1

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == "":
        if predicted_text == "":
            return 0
        return 1

    target_words = target_text.split(" ")

    return editdistance.eval(target_words, predicted_text.split(" ")) / len(target_words)
