import sys
from os.path import dirname, abspath, join

ROOT  = dirname(dirname(abspath(__file__)))
sys.path.append(join(ROOT))
SRC   = join(ROOT, 'src')
MODEL = join(ROOT, 'models')
DATA  = join(ROOT, 'data')
SCRIPTS = join(ROOT, 'scripts')
RESULTS = join(ROOT, 'results')
NOTEBOOK = join(ROOT, 'notebooks')
FEATURES = join(ROOT, 'features')
RESULTS  = join(ROOT, 'results')
IMAGES   = join(ROOT, 'images')
CLASSIFIER = join(ROOT, 'classifier')
CSV = join(ROOT, 'CSV')

LATENT_HATRED_FOLDER = join(DATA, "latent-hatred")
TOXICN_FOLDER = join(DATA, "toxicn")

RES_LLM_ENC = join(RESULTS, 'llm-as-encoder')
RES_LLM_PRO = join(RESULTS, 'llm-with-prompt')

# model mapping
MODEL_MAPPING = {
    "dolly-v2-3b"    : "dolly_v2",
    "dolly-v2-7b"    : "dolly_v2",#!
    "dolly-v2-12b"   : "dolly_v2",#!
    "vicuna-7b-v1.3" : "vicuna", #!
    "vicuna-13b-v1.3": "vicuna",#!
    "stablelm-tuned-alpha-3b": "stablelm", #!
    "stablelm-tuned-alpha-7b": "stablelm", #!
    "RedPajama-INCITE-Instruct-3B-v1": "redpajama-incite",
    "Baichuan-7B"  : "baichuan",
    "Baichuan-13B-Chat"  : "baichuan",
    "open_llama_3b"  : "open-llama",
    "open_llama_7b"  : "open-llama",
    "open_llama_13b" : "open-llama",
    "Llama-2-7b-chat-hf"  : "llama-2",
    "Llama-2-13b-chat-hf" : "llama-2",
    "Llama2-Chinese-7b-Chat" : "llama2-chinese",
    "Qwen-7B-Chat" : "Qwen",
}


CODETYPES = {
    "en": [
        "abbreviation. An abbreviation is a shortened form of a word or phrase, by any method.",
        "metaphor. A metaphor is a figure of speech that, for rhetorical effect, directly refers to one thing by mentioning another.",
        "irony. Irony is the juxtaposition of what on the surface appears to be the case and what is actually the case or to be expected。",
        "pun. A pun, also rarely known as paronomasia, is a form of word play that exploits multiple meanings of a term, or of similar-sounding words, for an intended humorous or rhetorical effect.",
        "idiom. A Slang is a vocabulary of an informal register, common in verbal conversation. It also sometimes refers to the language generally exclusive to the members of particular in-groups in order to establish group identity, exclude outsiders, or both. ",
        "argot. An argot is a language used by various groups to prevent outsiders from understanding their conversations.",
        "others. Those not belonging to the above coding types are classified as others"
        ],
    "zh": [
        "缩写。缩写是在使用拼音文字的语言中，对于常用的词组（多为专名）以及少数常用的词所采用的简便写法。", 
        "隐喻。隐喻是用在比较两个无关事物，制造的一个修辞的转义。", 
        "反讽。反讽的特征是字面意义与深层意义不一致，即言在此而意在彼。", 
        "双关。双关是一种修辞方法，利用谐音或借义，使一个句子可以理解成多种意思。", 
        "俗语。俚语是指民间非正式且较口语的词句，又叫方言，土话，民间非官方的腔调和语言习惯。", 
        "黑话。黑话指流行于某一特殊行业中，非局外人所能了解的语言。在互联网时代，网络流行语是网民间的黑话。", # include online slangs
        "其他。不属于以上编码类型的为其他类。",
        ]
    }

#! original codetype information
# CODETYPES = {
#     "en": [
#         "The abbreviation 'txl' used in the sentence 'Txl is a social issue, someone should manage it, not treating it as invisible' stands for homosexuality.",
#         "In the statement, 'Debuff is maxed out, I'm a Uighur, far-left, Puxin male, pink nationalist maggot studying in Xi'an', the speaker uses the term 'maggot' metaphorically to belittle Uighur people.",
#         "In the phrase 'Surely no one really wants to marry a little fairy, right?', 'little fairy' is an ironic usage of a term originally intended as a compliment but is used here to derogatively refer to women.",
#         "In the sentence 'Can a feminist man withstand 414?', the number '414' resembles the pronunciation of the Chinese phrase 'to die' or 'dead serious'",
#         "The term 'gay' in the question 'So can a gay man accept making love with a woman?' was first popularized as a reference to male homosexuality in Hong Kong.",
#         "The sentence 'I will embrace dogs, but not black people' uses parallelism to convey that the speaker's affection for dogs exceeds that for black people.",
#         "In the statement 'It's okay, aren't there a lot of lick-dogs around?', 'lick-dog' is a popular internet slang, usually used mockingly to describe men who give unrequited emotional support."
#         ],
#     "zh": [
#         "“txl是社会问题，应该有人管，而不是当透明人”这句话中的txl是同性恋的缩写" ,
#         "“debuff叠满了，本人西安上学维吾尔族极左普信男粉红爱国蛆”这句话中说话者用蛆来比喻维吾尔族人" ,
#         "“不会真有人想和小仙女结婚吧不会吧不会吧”这句话中的小仙女原是褒义词但此处用反语表达对女性的贬义性称呼" ,
#         "“女拳男能不能414阿？”这句话中的414和中文的死一死发音很像" ,
#         "“那么基佬能接受和女人make love吗？”这句话中的“基佬”是香港地区最早流行的对男同性恋的说法" ,
#         "“我会抱狗 但是不会抱黑人”这句话用并列句式传达出说话者对狗的热爱程度大于黑人" ,
#         "“没事，不是说舔狗很多吗？”这句话中的舔狗是一种网络流行用语，通常用来是嘲讽不计回报为情付出的男生" 
#         ]
#     }
