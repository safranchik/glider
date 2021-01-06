from .labeling_function import LabelingFunction


class PhraseMatcherLF(LabelingFunction):

    def __init__(self, phrase_list, label, name, lowercase=True):
        """
        :param phrase_list: list of strings containing phrases to match tokens with
        :param label: label to cast when tokens match keywords
        :param name: name of labeling function
        :param lowercase: indicates whether to do case-sensitive phrase matching
        """

        self.phrase_set = {tuple(phrase.split()) for phrase in phrase_list}  # partitions phrases by space
        self.label = label
        self.name = name
        self.lowercase = lowercase

        self.max_phrase_length = max(len(phrase) for phrase in self.phrase_set)

    def get_name(self):
        return self.name

    def apply(self, sample, metadata):
        if self.lowercase:
            tokens = [t.lower() for t in metadata['tokens']]
        else:
            tokens = metadata['tokens']

        for i in range(len(tokens) - 1):
            for j in range(i + 1, min(len(tokens) + 1, i + self.max_phrase_length + 1)):
                if tuple(tokens[i:j]) in self.phrase_set:
                    return self.label
