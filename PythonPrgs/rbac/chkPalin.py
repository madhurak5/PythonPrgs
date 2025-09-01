def chk_palin(word, sentence): ->bool:
    """ This function returns true if the reverse of the word is present in the sentence

    For example:
    chk_palin("madam", "madam inspires me")
    True
    chk_palin("india", "india is my country")
    False
    chk_palin("gadag", "one of the districts in Karnataka is gadag")
    True
    """
    return word[::-1] in sentence