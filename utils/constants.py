# DO NOT CHANGE! These are backup numbers for re-generation of a particular setup. Do not change the values
# until this notice is removed.

# The user id column number in the revision history tsv
USER_ID = 3
# The user name column number in the revision history tsv
USER_NAME = 4
# The page id column number in the revision history tsv
PAGE_ID = 1
# The lower bound for user edits/distinct edits (depends on USER_FILTER_METHOD), anyone below is filtered out
# Down from 40
USER_INACTIVITY_THRESH = 20
# The upper bound for users, anyone above that number of edits is treated as a bot.
USER_BOT_THRESH = 200000
# For percentile-based document filtering. Only those in the top 1 - DOC_PERCENTILE percent are kept.
DOC_PERCENTILE = 0.9
# The lower bound for docs, anything below is filtered out.
# Down from 75 (damn that was high)
DOC_INACTIVITY_THRESH = 20
# The minimum size of a non-stub document. Anything shorter is filtered out.
# Down from 500, now only filtering out the absolute garbage.
DOC_STUB_THRESH = 100
# Doc filtering method:
# 'value' : By raw edit count
# 'value_bin' : By distinct edit count (i.e. distinct users)
# 'percentile' : By percentile
DOC_FILTER_METHOD = 'value'
# User filtering method. Similar to doc filtering method, doesn't have the 'percentile' method though.
USER_FILTER_METHOD = 'value'
# Upper relative bound for doc frequency of a term. Any term with doc frequency higher than
# DOC_FREQ_UPPER_REL * number of docs will be filtered out as a stopword.
# Up from 0.1, in order not to discard important words (I mean, who knows).
DOC_FREQ_UPPER_REL = 0.1
# Lower absolute bound for doc frequency of a term. For filtering out overly rare terms (which might be due to typos).
DOC_FREQ_LOWER_ABS = 50
# Whether to discard list-like docs or not
DISCARD_LISTS = True
# Whether to discard yearly event docs or not
DISCARD_YEAR_LISTS = True
DEFAULT_HOLDOUT_SIZE = 20