library(shiny)
library(DT)
library(ggplot2)
library(plotly)
library(shinythemes)
library(shinydashboard)
library(shinymanager)


# load your dependencies
files_sources = list.files('./R', full.names = T)
sapply(files_sources, source)
date_stamp = '20200219'

# inactivity script
inactivity <- "
function idleTimer() {
  var t = setTimeout(logout, 120000);
  window.onmousemove = resetTimer; // catches mouse movements
  window.onmousedown = resetTimer; // catches mouse movements
  window.onclick = resetTimer;     // catches mouse clicks
  window.onscroll = resetTimer;    // catches scrolling
  window.onkeypress = resetTimer;  //catches keyboard actions
  
  function logout() {
    window.close();  //close the window
  }
  
  function resetTimer() {
    clearTimeout(t);
    t = setTimeout(logout, 120000);  // time is in milliseconds (1000 is 1 second)
  }
}
idleTimer();"


header <-
    dashboardHeader(
        title = 'AI-Trainer',
        #titleWidth = 350,
        dropdownMenu(type = "tasks", badgeStatus = "success",
                     taskItem(value = 98, color = "green",
                              "Files to classifiy"
                     )
                     #,
                     #taskItem(value = 17, color = "aqua",
                     #         "Project X"
                     #),
                     #taskItem(value = 75, color = "yellow",
                     #         "Server deployment"
                     #),
                     #taskItem(value = 80, color = "red",
                     #         "Overall project"
                     #)
        ),
        dropdownMenu(
            type = "notifications",
            icon = icon("question-circle"),
            badgeStatus = NULL
        )
    )

sidebar <- 
    dashboardSidebar(
        width = 350,
        selectInput(
            selectize = T,
            label = 'Select date',
            'date_stamp',
            choices = list.dirs('../data/resmed/classification', recursive = F, full.names = F)
        ),
        textOutput(
          outputId = 'current_file'
        ),
        uiOutput("category"),
        br(),
        actionButton(
            width = '87%',
            inputId = "btn_normal_series",
            label = "Normal sample",
            icon = icon('fas fa-check')
        ),
        actionButton(
            width = '87%',
            inputId = "btn_abnormal_series",
            label = "Abnormal sample",
            icon = icon('fas fa-times')
        )
        #,
        #br(),
        #textOutput(
        #  outputId = 'current_file'
        #)
        #,
        #textOutput(
        #  outputId = 'pct_anomaly'
        #),
        #textOutput(
        #  outputId = 'pct_normal'
        #)
    )

body <- dashboardBody(
    #tags$head(tags$style(
    #    HTML('.wrapper {height: auto !important; position:relative; overflow-x:hidden; overflow-y:hidden}')
    #)),
    fluidRow(
        tabBox(
            title = "Respiration Analysis",
            width = 12,
            height = 700,
            tabPanel(
                'Overview',
                plotlyOutput(
                    outputId = "plot_overview"
                )
            )
            #,
            #tabPanel(
            #    'Raw data',
            #    actionButton(
            #      width = '120',
            #      inputId = "btn_download_raw",
            #      label = "Download",
            #      icon = icon('fas fa-download')
            #    ),
            #    dataTableOutput(
            #        outputId = "raw_df"
            #    )
            #)
        )
    )
    #,
    #fluidRow(
    #    box(
    #        title = 'Your classifications',
    #        width = 12,
    #        actionButton(
    #            width = '120',
    #            inputId = "btn_delete_row",
    #            label = "Delete Row!",
    #            icon = icon('fas fa-trash-alt')
    #        )
    #        #,
    #        #dataTableOutput(
    #        #    height = '15%',
    #        #    outputId = "classification_df")
    #    )
    #)
)

secure_app(head_auth = tags$script(inactivity),
    dashboardPage(
        header,
        sidebar,
        body, 
        skin = "green"
    )
)
