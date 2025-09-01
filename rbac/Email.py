import smtplib, ssl
def sendmailto(useremail):
    port = 465
    sndMail = "madhurask375@gmail.com"
    recMail = useremail
    pwd = input("Password please :")
    msg = "Test msg being sent using Python"
    ctx = ssl.create_default_context()
    server1 = smtplib.SMTP("smtp.gmail.com", 587)
    server1.starttls()
    server1.login(sndMail, pwd)
    print("login success ")
    server1.sendmail(sndMail, recMail, msg)
    print("Email sent to ", recMail)
