import dash
import dash_html_components as html


def build_header(app: dash.Dash):
    header = html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("tvd.svg"),
                        id="plotly-image",
                        style={
                            "height": "40px",
                            "width": "auto",
                            "margin-bottom": "25px",
                        },
                    )
                ],
                className="one-third column",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Atemteurer",
                                style={"margin-bottom": "0px"},
                            ),
                            html.H5(
                                "AiExplainer", style={"margin-top": "0px"}
                            ),
                        ]
                    )
                ],
                className="one-half column",
                id="title",
            ),
            html.Div(
                [
                    html.A(
                        html.Button("Learn More", id="learn-more-button"),
                        href="https://www.trivadis.com/de/big-data-science",
                    )
                ],
                className="one-third column",
                id="button",
            ),
        ],
        id="header",
        className="row flex-display",
        style={"margin-bottom": "25px"},
    )

    return header
