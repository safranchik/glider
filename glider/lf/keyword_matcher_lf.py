from .labeling_function import LabelingFunction


class KeywordMatcherLF(LabelingFunction):
    """
    Labeling function that casts votes based on keyword matches
    """

    def __init__(self, keyword_set, label, name, lowercase=True):
        """
        :param keyword_set: set of keywords used to match tokens with
        :param label: label to cast when tokens match keywords
        :param name: name of labeling function
        :param lowercase: indicates whether to do case-sensitive keyword matching
        """

        self.keyword_set = keyword_set
        self.label = label
        self.name = name
        self.lowercase = lowercase

    def get_name(self):
        return self.name

    def apply(self, context):
        if self.lowercase:
            tokens = {t.lower() for t in context}
        else:
            tokens = set(context)

        if len(self.keyword_set.intersection(tokens)):
            return self.label
