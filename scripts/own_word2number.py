#%% 
def own_word2num(string):
    getal_dictionary = {
        'nul': 0,
        'half':0.5,
        'een': 1,
        'twee': 2,
        'drie': 3,
        'vier': 4,
        'vijf': 5,
        'zes': 6,
        'zeven': 7,
        'acht': 8,
        'negen': 9,
        'tien': 10,
        'elf': 11,
        'twaalf': 12,
        'dertien': 13,
        'veertien': 14,
        'vijftien': 15,
        'zestien': 16,
        'zeventien': 17,
        'achttien': 18,
        'negentien': 19,
        'twintig': 20,
        'eenentwintig': 21,
        'tweeentwintig': 22,
        'drieentwintig': 23,
        'vierentwintig': 24,
        'vijfentwintig': 25,
        'zessentwintig': 26,
        'zevenentwintig': 27,
        'achtentwintig': 28,
        'negenentwintig': 29,
        'dertig': 30,
        'eenendertig': 31,
        'tweeendertig': 32,
        'drieendertig': 33,
        'vierendertig': 34,
        'vijfendertig': 35,
        'zessendertig': 36,
        'zevenendertig': 37,
        'achtendertig': 38,
        'negenendertig': 39,
        'veertig': 40,
        'eenenveertig': 41,
        'tweeenveertig': 42,
        'drieenveertig': 43,
        'vierenveertig': 44,
        'vijfenveertig': 45,
        'zessenveertig': 46,
        'zevenenveertig': 47,
        'achtenveertig': 48,
        'negenenveertig': 49,
        'vijftig': 50,
        'eenenvijftig': 51,
        'tweeenvijftig': 52,
        'drieenvijftig': 53,
        'vierenvijftig': 54,
        'vijfenvijftig': 55,
        'zessenvijftig': 56,
        'zevenenvijftig': 57,
        'achtenvijftig': 58,
        'negenenvijftig': 59,
        'zestig': 60,
        'eenenzestig': 61,
        'tweeenzestig': 62,
        'drieenzestig': 63,
        'vierenzestig': 64,
        'vijfenzestig': 65,
        'zessenzestig': 66,
        'zevenenzestig': 67,
        'achtenzestig': 68,
        'negenenzestig': 69,
        'zeventig': 70,
        'eenenzeventig': 71,
        'tweeenzeventig': 72,
        'drieenzeventig': 73,
        'vierenzeventig': 74,
        'vijfenzeventig': 75,
        'zessenzeventig': 76,
        'zevenenzeventig': 77,
        'achtenzeventig': 78,
        'negenenzeventig': 79,
        'tachtig': 80,
        'eenentachtig': 81,
        'tweeentachtig': 82,
        'drieentachtig': 83,
        'vierentachtig': 84,
        'vijfentachtig': 85,
        'zessentachtig': 86,
        'zevenentachtig': 87,
        'achtentachtig': 88,
        'negenentachtig': 89,
        'negentig': 90,
        'eenennegentig': 91,
        'tweeennegentig': 92,
        'drieennegentig': 93,
        'vierennegentig': 94,
        'vijfennegentig': 95,
        'zessennegentig': 96,
        'zevenennegentig': 97,
        'achtennegentig': 98,
        'negenennegentig': 99,
        'honderd': 100,
        'duizend': 1000,
        'miljoen': 1000000,
        'miljard': 1000000000,
        'punt': '.'
    }

    return(getal_dictionary[string])