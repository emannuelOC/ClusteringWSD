# -*- coding: utf-8 -*-
"""
    *
   * *
  * S * 
 * D G *
* * * * *

Emannuel Fernandes de Oliveira Carvalho

Text-processing
"""

# preparing text
def prepareText(rawText):
    """Returns a list of strings containing
    the words in the text, without ponctuation"""
    newText = rawText.split();          # String to list
    ponctuation = '.,::\'"?!-/()[]{}'   # Taking ponctuation away
    newText = [w.strip(ponctuation) for w in newText]
    newText = [w.lower() for w in newText if w != " "]
    return newText
    
# add methods for striping html or xml tags 