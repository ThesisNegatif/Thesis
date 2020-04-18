import nltk
from datetime import datetime
from sqlalchemy.orm import object_session
from rake_nltk import Rake

# Below import isn't from this directory, but integrates directly into Quill Project codebase
from .. import models

__all__ = [
    'Keyworder',
    'MNelsonKeyworder'
]
class Keyworder(object):
    stop_word_additions = set()
    auto_name = "Standard"
    def this_filter(self, text, top=100):
        r = Rake()
        r.extract_keywords_from_text(text)
        phrases = r.get_ranked_phrases()
        most_common = phrases[0:top]
        return most_common

    def get_event_aspects(self, event):
        if hasattr(event, 'proposal_name'):
            return str(event.proposal_name) + '\n' + str(event.description)
        else:
            return str(event.description)

    def keyword_convention(self, convention_object, reference_texts,
                            keyword_pack=None,
                            blank_existing_keywords=False,
                            add_to_master_approved_list=True,
                            debug=False, top=100):
        this_session = object_session(convention_object)
        if not keyword_pack:
            keyword_pack = models.KeywordPack(name='%s: %s' % (self.auto_name,  datetime.now().ctime()))
            this_session.add(keyword_pack)
        kws = set()
        for reference_text in reference_texts:
            kws = kws.union(set(self.this_filter(reference_text, top=top)))

        string_to_kw = {}
        for this_word in kws:
            kw = keyword_pack.add_approved_keyword(this_word.title())
            string_to_kw[this_word] = kw
            if add_to_master_approved_list:
                convention_object.approved_keywords.add(kw)

        if debug:
            print("Approved Keywords")
            print("=================")
            for s in string_to_kw:
                print(s)

        all_events = this_session.query(models.Event) \
                            .join(models.Session) \
                            .join(models.Committee) \
                            .join(models.Convention) \
                            .filter(models.Convention.id==convention_object.id) \
                            .all()
        for e in all_events:
            if blank_existing_keywords:
                e.keywords = []
            if debug:
                print("Working on %s " % (e,))
            keywords = self.this_filter(self.get_event_aspects(e), top=top)
            for k in keywords:
                if k in string_to_kw:
                    if string_to_kw[k] in e.keywords:
                        continue
                    else:
                        e.keywords.append(string_to_kw[k])
                        if debug:
                            print(" ---- added keyword %s" % (k.title(), ))

    def keyword_from_draft_text(self, convention_object, reference_text_moment, *args, **keywords):
        reference_text = reference_text_moment.get_draft_document_text()
        return self.keyword_convention(convention_object, [reference_text,], *args, **keywords)


class MNelsonKeyworder(Keyworder):
    stop_word_additions = set(['Constitute',
            'session',
            'cause',
            'hold',
            'year',
            'next',
            'said',
            'article',
            'section',
            'general',
            'sec',
            'used',
            'thereof',
            'manner',
            'held'])
