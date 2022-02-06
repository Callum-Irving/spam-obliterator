import random
import pandas
import string
import numpy as np
from pandas import DataFrame, Series

BIAS = -1


def read_data_csv(file: str) -> DataFrame:
    """
    Parse spam sms message dataset from a csv file.

    Takes a file containing the csv data and returns a pandas dataframe.
    """

    try:
        csv = pandas.read_csv(file, encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"ERROR: Could not open file: {file}")
        exit(1)

    # Set column names
    csv = csv[["v1", "v2"]]
    csv.rename(columns={"v1": "spam", "v2": "text"}, inplace=True)

    # Convert spam column to boolean values
    csv.spam = csv.spam.apply(lambda x: x == "spam")

    return csv


def clean_data(data: Series) -> Series:
    """
    Clean up dataset.

    Takes a pandas Series and applies normalizing operations to it.
    """

    cleaned = data.apply(clean_message)
    assert isinstance(cleaned, Series)

    return cleaned


def clean_message(message: str) -> str:
    """
    Clean up a string.

    Converts all letters to lowercase, replaces '$' with 'dollar' and removes
    punctuation.
    """
    return (
        message.lower()
        .replace("$", " dollar ")
        .translate(str.maketrans("", "", string.punctuation))
    )


def create_bags_of_words(data: DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """
    Converts pandas DataFrame to 2 separate bags of words (spam and non-spam).

    Returns two dicts that map a word to the chance that it occurs given the
    message is spam or non-spam.
    """
    spam_words = " ".join(data[data.spam == True].text).split()
    ham_words = " ".join(data[data.spam == False].text).split()
    common_words = set(spam_words).intersection(set(ham_words))

    spam_probs = {}
    ham_probs = {}
    for w in common_words:
        spam_probs[w] = spam_words.count(w) / len(spam_words)
        ham_probs[w] = ham_words.count(w) / len(ham_words)

    return spam_probs, ham_probs


def predict_on_string(
    string: str,
    spam_frac: float,
    spam_probs: dict[str, float],
    ham_probs: dict[str, float],
) -> bool:
    words = clean_message(string).split()
    words = [w for w in words if w in spam_probs]

    spam_score = sum(list(map(np.log, [spam_probs[w] for w in words]))) + np.log(
        spam_frac
    )
    ham_score = sum(list(map(np.log, [ham_probs[w] for w in words]))) + np.log(
        1 - spam_frac
    )

    spam_score += BIAS
    return spam_score > ham_score


def predict_on_test_set(
    test_set: DataFrame,
    spam_frac: float,
    spam_probs: dict[str, float],
    ham_probs: dict[str, float],
) -> float:
    """
    Run classifier on test set.

    Returns computed F-score.
    """

    predictions = test.text.apply(
        lambda x: predict_on_string(x, spam_frac, spam_probs, ham_probs)
    )

    tp = np.sum((predictions == True) & (test.spam == True))
    fp = np.sum((predictions == True) & (test.spam == False))
    fn = np.sum((predictions == False) & (test.spam == True))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


# TODO: Make model a class
if __name__ == "__main__":
    df = read_data_csv("data.csv")

    # Split into train and test set
    train = df.sample(frac=0.8, random_state=200)
    assert isinstance(train, DataFrame)
    test = df.drop(train.index)
    assert isinstance(test, DataFrame)

    # Train the classifier
    spam_frac = train.spam.mean()
    train.text = clean_data(train.text)
    spam_probs, ham_probs = create_bags_of_words(train)

    f_scores = []
    for b in range(-10, 10):
        BIAS = b
        f_scores.append(predict_on_test_set(test, spam_frac, spam_probs, ham_probs))
        print("Running with bias", BIAS, ":", f_scores[-1])

    print(max(f_scores))
