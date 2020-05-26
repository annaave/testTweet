import PyPDF2
import pandas as pd


class ReadText:
    def __init__(self, file_name):
        self.file_name = file_name

    def openReadSave(self):
        # open allows you to read the file.
        pdfFileObj = open(self.file_name, 'rb')

        # The pdfReader variable is a readable object that will be parsed.
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        # Discerning the number of pages will allow us to parse through all the pages.
        num_pages = pdfReader.numPages
        count = 0
        text = ""

        # The while loop will read each page.
        while count < num_pages:
            pageObj = pdfReader.getPage(count)
            count += 1
            text += pageObj.extractText()

        text = text.split('\n')
        text_df = pd.DataFrame(text)
        print(text_df)
        text_df.to_csv('eng_human_rights.csv', index=False)

