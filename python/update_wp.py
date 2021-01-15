from wordpress_xmlrpc import Client
from wordpress_xmlrpc.methods import posts
from wordpress_xmlrpc import WordPressPage
import pandas as pd
import numpy as np
from python.Level1.Level2.predict import normal_round

INDICATOR_DATA_FILE = 'data/indicator_data.csv'
PORTFOLIO_HISTORY_FILE = 'data/portfolio_history/aborger@nnu.edu.csv'


def upload_indicator():
    # Get stock info
    data = pd.read_csv(INDICATOR_DATA_FILE, sep=r'\s*,\s*', engine='python')
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


def upload_performance():
    # Uses chart from canvasjs.com
    # Documentation: https://canvasjs.com/docs/charts/basics-of-creating-html5-chart/
    data = pd.read_csv(PORTFOLIO_HISTORY_FILE, sep=r'\s*,\s*', engine='python')
    keys = data.keys()
    np_data=data.to_numpy()

    # Convert to data into html line graph

    
    string = '<head>'
    string += '<script type="text/javascript">'
    string += 'window.onload = function () {'
    string += 'var chart = new CanvasJS.Chart("chartContainer", {'
    string += 'title:{ text: "Kara $10,000 Account 1Y" },'
    string += 'axisY: { title: "Equity ($)", gridColor: "#B5B5B5" },'
    string += 'data: [{ type: "line", color: "#3DA37B",'
    string += 'dataPoints: ['

    # Data
    for row in np_data:
        date = row[0].replace('-', ', ')
        string += '{ x: new Date(' + date + '), y: ' + str(row[1]) + ' },'

    string = string[:-1]
    string += ']}]});'
    string += 'chart.render();'
    string += '}'
    string += '</script>'
    string += '<script type="text/javascript" src="https://canvasjs.com/assets/script/canvasjs.min.js"></script>'
    string += '</head>'
    string += '<body>'
    string += '<div id="chartContainer" style="height: 300px; width: 100%;"></div>'
    string += '</body>'

    client = Client('http://karatrader.com/xmlrpc.php', 'karatrader', 'N5xXLsSZzaD3pz7l')

    
    page = WordPressPage()
    page.title = 'Kara Origin'
    page.content = string
    page.post_status = 'publish'
    page.id = 618
    client.call(posts.EditPost(page.id, page))
    