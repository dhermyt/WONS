## ![logo](http://wons.net.pl/assets/images/wons_icon.png) [wons.net.pl](http://wons.net.pl)

# What is WONS?

*WONS* a.k.a *Word Operating Neural System* is a public API for text analysis in Polish language.

# Feature list

Completed:

- word synonyms

Under development:

- sentiment analysis - checks whether the comment is positive, neutral or negative

Planned:

- lemmatizer - get a root form of a word
- topic tagging - generates keywords for articles

# How to use it

WONS is automatically deployed from release branch to [wons.net.pl](http://wons.net.pl)

Check this URL for API reference and usage examples.

# Running locally

To run WONS locally make sure you installed all of the packages from requirements.txt
Next download [Data](https://app.box.com/v/wons-data) package and extract to the root folder of the application.
WONS has a Flask API under web/api.py, so run it and feel free to use it on your local machine!

# License

Source code is distributed under BSD 2-Clause License (see [LICENSE.md](LICENSE.md))

The following projects are used to make it work:

- [Słowosieć](http://plwordnet.pwr.wroc.pl/wordnet/) ([License](http://nlp.pwr.wroc.pl/plwordnet/license/))
- [PoliMorf](http://zil.ipipan.waw.pl/PoliMorf) (2-clause BSD License)


