library(shiny)
library(ggplot2)
library(dplyr)
library(plotly)
library(tidyr)
library(shinyalert)
library(shinymanager)


# load your dependencies
files_sources = list.files('./R', full.names = T)
sapply(files_sources, source)


# data.frame with credentials info
credentials <- data.frame(
    user = c("yves"),
    password = c("yves"),
    # comment = c("alsace", "auvergne", "bretagne"), %>% 
    stringsAsFactors = FALSE
)


shinyServer(function(input, output) {
    
    # secure tocken
    result_auth <- secure_server(check_credentials = check_credentials(credentials))
    
    ######################################################################
    # Reactive values
    ######################################################################

    # dataframe with classifications
    user_classification_df = reactiveVal(
        data_frame(
            uid = as.character(),
            station = as.character(),
            window_start = as.character(),
            window_end = as.character(),
            is_anomaly = as.integer(),
            class = as.character()
        )
    )
    
    # data frame placeholder for raw df
    raw_df <- reactiveVal(
        data_frame()
    )
    
    ######################################################################
    # selectionList controls
    ######################################################################
    
    #---------------------------------------------------------------------
    # Update raw data variable when input file changes
    
    observeEvent(input$file_name, {
        raw_df(read_raw_data(input$file_name))
    })
    
    
    ######################################################################
    # buttons
    ######################################################################
    
    #---------------------------------------------------------------------
    # sidebar
    
    # create or update a line in user_classification df
    # that assigns a posisitive detection label to the active sample.
    observeEvent(input$btn_correct_classification, {
        
        u_id <- sub('\\..*$', '', basename(input$file_name))
        
        updated_df <- update_classification_df(
            class_df = user_classification_df(), 
            raw_df = raw_df(),
            u_id = u_id,
            is_correct = T
        )
        
        user_classification_df(updated_df)
    })
    
    # create or update a line in user_classification df
    # that assigns a negative detection label to the active sample.
    observeEvent(input$btn_wrong_classification, {
        
        u_id <- sub('\\..*$', '', basename(input$file_name))
        
        updated_df <- update_classification_df(
            class_df = user_classification_df(), 
            raw_df = raw_df(),
            u_id = u_id,
            is_correct = F
        )
        
        user_classification_df(updated_df)
    })
    
    # that assigns a posisitive detection label to the active sample.
    observeEvent(input$btn_save_classification, {
        
        storage_write_file(user_classification_df())
        
        shinyalert::shinyalert("Success", "File has been transfered to Azure.", type = "success")
    })
    
    
    #---------------------------------------------------------------------
    # User classification box
    
    # create or update a line in user_classification df
    # that assigns a posisitive detection label to the active sample.
    observeEvent(input$btn_delete_row, {
        new_df = user_classification_df()
        
        if (!is.null(input$classification_df_rows_selected)) {
            new_df <- new_df[-as.numeric(input$classification_df_rows_selected),]
        }
        user_classification_df(new_df)
    })
    
    
    ######################################################################
    # Plots
    ######################################################################
    
    #---------------------------------------------------------------------
    # Respiration Analysis tabBox > OVerview
    
    # Overview with all variables
    output$plot_overview <- renderPlotly({
        
        # get preporcessed input
        pp_df <- preprocess_data(raw_df = raw_df())
        
        # create labels
        breaks <-
            seq.int(from = min(ceiling(pp_df$date_numeric)),
                    to = max(floor(pp_df$date_numeric)),
                    by = 5)
        break_label <-
            as.POSIXct(breaks, origin = "1970-01-01", tz = 'CET')  
        
        # create plot
        p <-  ggplot(pp_df, aes(x = date_numeric, y = value)) +
            geom_line() +
            geom_point(size = .5,
                       aes(text = sprintf(
                           "%s<br>Value: %s", date_string, value
                       ))) +
            scale_x_continuous(
                breaks = breaks,
                labels = break_label,
                limits = c(head(breaks, 1), tail(breaks, 1))
            ) +
            facet_wrap( ~ category, ncol = 1, scales = 'free_y') +
            theme_bw() +
            theme(
                axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                text = element_text(size = 16)
            )
        
        p <- ggplotly(p, tooltip = c('text')) %>% layout(height = 630)
        
        return(p)
    })
    
    ######################################################################
    # Data tables
    ######################################################################
    
    
    #---------------------------------------------------------------------
    # Respiration Analysis tabBox > Raw Data
    
    # shows the raw data of the loaded file
    output$raw_df = DT::renderDataTable({
        raw_df() %>% 
            dplyr::mutate(
                date = format(date, format = '%Y-%m-%d %H:%M:%OS4')
            ) %>% 
            tidyr::spread(
                key = category,
                value = value
            )
    }, options = list(pageLength = 15))
    
    #---------------------------------------------------------------------
    # Your classifications tabBox > Data table
    
    # shows the classification that the user made
    output$classification_df <- renderDataTable({
        datatable(
            user_classification_df(), 
            options = list(dom = 'ft'))
    })
})
