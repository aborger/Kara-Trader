from wordpress_xmlrpc import Client
from wordpress_xmlrpc.methods import posts
from wordpress_xmlrpc import WordPressPage
import pandas as pd
import numpy as np
from python.Level1.Level2.predict import normal_round


def upload(DATA_FILE):
    # Get stock info
    data = pd.read_csv(DATA_FILE, sep=r'\s*,\s*', engine='python')
    keys = data.keys()
    sorted_data = data.sort_values(by=['Predicted Gain'], ascending=False)
    np_data = sorted_data.to_numpy()
    # convert to strings
    for row in np_data:
        row[1] = str(normal_round(row[1], 2)) + '%'

    # Convert data into html table
    string = '<table class="indicator-table">'

    # Make Header
    string += '<tr>'
    for col in keys:
        string += '<th class ="indicator-th">' + col + '</th>'
    string += '</tr>'

    # Insert rest of data
    for row in np_data:
        string += '<tr>'
        for col in row:
            string += '<td class = "indicator-td">' + str(col) + '</td>'
        string += '</tr>'

    # Close table tag
    string += '</table>'

    print('uploading!')

    # Update website
    client = Client('http://karatrader.com/xmlrpc.php', 'karatrader', 'N5xXLsSZzaD3pz7l')

    page = WordPressPage()
    page.title = 'Indicator Stats'
    page.content = string
    page.post_status = 'publish'
    page.id = 739
    client.call(posts.EditPost(page.id, page))
