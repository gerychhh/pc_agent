from docx import Document

doc = Document()
doc.add_paragraph("Я - документ Word, созданный на рабочем столе.")
doc.save("$desktop\о себе.docx")