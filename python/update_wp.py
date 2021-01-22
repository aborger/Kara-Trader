from wordpress_xmlrpc import Client
from wordpress_xmlrpc.methods import posts
from wordpress_xmlrpc import WordPressPage
import pandas as pd
import numpy as np
from python.Level1.Level2.predict import normal_round

INDICATOR_DATA_FILE = 'data/indicator_data.csv'
PORTFOLIO_HISTORY_FILE = 'data/portfolio_history/aborger@nnu.edu.csv'
STOCK_HISTORY_FILE = 'data/stock_history/SPY.csv'
WORDPRESS_DATA_FILE = '../Sensitive_Data/wordpress.csv'


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

    wordpress_data = pd.read_csv(WORDPRESS_DATA_FILE, sep=r'\s*,\s*', engine='python')
    wordpress_data = wordpress_data.iloc[0]

    # Update website
    try:
        client = Client('http://karatrader.com/xmlrpc.php', wordpress_data["username"], wordpress_data["password"])
    except wordpress_xmlrpc.exceptions.ServerConnectionError:
        print('Error, did not upload!')

    page = WordPressPage()
    page.title = 'Indicator Stats'
    page.content = string
    page.post_status = 'publish'
    page.id = 739
    client.call(posts.EditPost(page.id, page))


def upload_performance():
    # Uses chart from canvasjs.com
    # Documentation: https://canvasjs.com/docs/charts/basics-of-creating-html5-chart/

    # collect kara data
    kara_data = pd.read_csv(PORTFOLIO_HISTORY_FILE, sep=r'\s*,\s*', engine='python')
    kara_np_data = kara_data.to_numpy()

    # collect s&p data
    sp_data = pd.read_csv(STOCK_HISTORY_FILE, sep=r'\s*,\s*', engine='python')
    sp_np_data = sp_data.to_numpy()

    num_shares = kara_np_data[0][1] / sp_np_data[0][1] # calculates number of spy shares to buy so both start of with same value
    # Convert to data into html line graph
    string = '<head>'
    string += '<script type="text/javascript">'
    string += 'window.onload = function () {'
    string += 'var chart = new CanvasJS.Chart("chartContainer", {'
    string += 'title:{ text: "Kara $10,000 Account 1Y" },'
    string += 'axisY: { title: "Equity ($)", gridColor: "#B5B5B5" },'
    string += 'data: ['

    # Kara Line
    string += '{ type: "line", color: "#3DA37B", showInLegend: true, legendText: "Kara Origin", dataPoints: ['
    # Kara Line Data

    for row in kara_np_data:
        date = row[0].replace('-', ', ')
        string += '{ x: new Date(' + date + '), y: ' + str(row[1]) + ' },'

    string = string[:-1]
    string += ']},'
    
    # S&P Line
    string += '{ type: "line", color: "#A9A9A9", showInLegend: true, legendText: "S&P 500", dataPoints: ['
    # S&P Line Data
    
    for row in sp_np_data:
        date = row[0][:10]
        date = date.replace('-', ', ')
        string += '{ x: new Date(' + date + '), y: ' + str(normal_round(row[1]*num_shares, 2)) + ' },'


    string = string[:-1]
    string += ']}'

    string += ']});'
    string += 'chart.render();'
    string += '}'
    string += '</script>'
    string += '<script type="text/javascript" src="https://canvasjs.com/assets/script/canvasjs.min.js"></script>'
    string += '</head>'
    string += '<body>'
    string += '<div id="chartContainer" style="height: 300px; width: 100%;"></div>'
    string += '</body>'


    try:
        client = Client('http://karatrader.com/xmlrpc.php', 'karatrader', 'N5xXLsSZzaD3pz7l')
    except wordpress_xmlrpc.exceptions.ServerConnectionError:
        print('Error, did not upload!')


    page = WordPressPage()
    page.title = 'Kara Origin'
    page.content = string
    page.post_status = 'publish'
    page.id = 618
    client.call(posts.EditPost(page.id, page))
    