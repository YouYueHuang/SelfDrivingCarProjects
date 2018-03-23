# calculate the count of each class in train, test, valid sets
y_train_elements, y_train_dist = np.unique(y_train, return_counts=True)
y_valid_elements, y_valid_dist = np.unique(y_valid, return_counts=True)
y_test_elements, y_test_dist = np.unique(y_test, return_counts=True)

def normalied_prob(data):
    return data/np.sum(data)

train = go.Bar(
    x = y_train_elements,
    y = y_train_dist,
    name='train',
    marker=dict(
        color='rgb(55, 83, 109)'
    )
)
valid = go.Bar(
    x = y_valid_elements,
    y = y_valid_dist,
    name='valid',
    marker=dict(
        color='rgb(26, 118, 255)'
    )
)
test = go.Bar(
    x = y_test_elements,
    y = y_test_dist,
    name='test',
    marker=dict(
        color='rgb(250, 118, 0)'
    )
)
data = [train, valid, test]
layout = go.Layout(
    title='Sign Class Distribution of Train, Test and Validation Sets',
    xaxis=dict(
        title='sign class',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='probability',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=1.0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='style-bar')