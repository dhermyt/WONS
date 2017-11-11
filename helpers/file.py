import codecs


class File:
    @staticmethod
    def append(filename, text):
        f = codecs.open(filename, 'a', 'utf-8')
        f.write(text)
        f.close()
