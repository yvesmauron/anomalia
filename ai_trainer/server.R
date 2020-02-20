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


wd = './'#C:/Users/yvesm/OneDrive - Trivadis AG/projects/atemreich/atemteurer/ai_trainer'

data_dir <- '../data/resmed/classification'
unclassified_path <- file.path(data_dir, date_stamp, 'unclassified')
normal_path <- file.path(data_dir, date_stamp, 'normal')
anomaly_path <- file.path(data_dir, date_stamp, 'anomaly')



#file_list <- list.files(file.path(data_dir, date_stamp, 'unclassified'))
#file_list_total <- list.files(file.path(data_dir, date_stamp), recursive = T)
#current_file <- file_id <- 1

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
    #user_classification_df = reactiveVal(
    #    data_frame(
    #        uid = as.character(),
    #        station = as.character(),
    #        window_start = as.character(),
    #        window_end = as.character(),
    #        is_anomaly = as.integer(),
    #        class = as.character()
    #    )
    #)
    
    # data frame placeholder for raw df
    file_id <- reactiveVal(
        1
    )
    
    file_list <- reactiveVal(
        c()
    )
    

    ######################################################################
    # selectionList controls
    ######################################################################
    
    #---------------------------------------------------------------------
    # Update raw data variable when input file changes
    
    observeEvent(input$date_stamp, {
        file_list(list.files(file.path(data_dir, input$date_stamp, 'unclassified')))
    })
    
    #observeEvent(input$date_stamp, {
    #    
    #})
    
    
    ######################################################################
    # buttons
    ######################################################################
    
    #---------------------------------------------------------------------
    # sidebar
    
    # create or update a line in user_classification df
    # that assigns a posisitive detection label to the active sample.
    observeEvent(input$btn_normal_series, {
        
        #u_id <- sub('\\..*$', '', basename(input$file_name))
        #
        #updated_df <- update_classification_df(
        #    class_df = user_classification_df(), 
        #    raw_df = raw_df(),
        #    u_id = u_id,
        #    is_correct = T
        #)
        #
        file.copy(from=file.path(wd, unclassified_path, file_list()[file_id()]), to=file.path(wd, normal_path, file_list()[file_id()]))
        file.remove(file.path(wd, unclassified_path, file_list()[file_id()]))
        #user_classification_df(updated_df)
        file_id(file_id() + 1)
    })
    
    # create or update a line in user_classification df
    # that assigns a negative detection label to the active sample.
    observeEvent(input$btn_abnormal_series, {
        
        #u_id <- sub('\\..*$', '', basename(input$file_name))
        #
        #updated_df <- update_classification_df(
        #    class_df = user_classification_df(), 
        #    raw_df = raw_df(),
        #    u_id = u_id,
        #    is_correct = F
        #)
        #
        #user_classification_df(updated_df)
        file.copy(from=file.path(wd, unclassified_path, file_list()[file_id()]), to=file.path(wd, anomaly_path, file_list()[file_id()]))
        file.remove(file.path(wd, unclassified_path, file_list()[file_id()]))
        file_id(file_id() + 1)
    })
    
    # that assigns a posisitive detection label to the active sample.
    #observeEvent(input$btn_save_classification, {
    #    
    #    storage_write_file(user_classification_df())
    #    
    #    shinyalert::shinyalert("Success", "File has been transfered to Azure.", type = "success")
    #})
    
    #---------------------------------------------------------------------
    # raw data
    
    # create or update a line in user_classification df
    # that assigns a posisitive detection label to the active sample.
    # Downloadable csv of selected dataset ----
    #output$downloa_raw_data <- downloadHandler(
    #    filename = function() {
    #        paste(input$file_name)
    #    },
    #    content = function(file) {
    #        write.csv(raw_data(), file, row.names = FALSE)
    #    }
    #)
    #
    
    #---------------------------------------------------------------------
    # User classification box
    
    # create or update a line in user_classification df
    # that assigns a posisitive detection label to the active sample.
    #observeEvent(input$btn_delete_row, {
    #    new_df = user_classification_df()
    #    
    #    if (!is.null(input$classification_df_rows_selected)) {
    #        new_df <- new_df[-as.numeric(input$classification_df_rows_selected),]
    #    }
    #    user_classification_df(new_df)
    #})
    
    ######################################################################
    # Text output
    ######################################################################
    
    #---------------------------------------------------------------------
    # Respiration Analysis tabBox > OVerview    
    output$current_file <- renderText(
        print(paste('Processing file:', file_list()[file_id()]))
    )
    
    #output$pct_processed <- renderText(
    #    print(paste('Pct processed:', 1 - 
    #                    (
    #                        length(list.files(unclassified_path, recursive = T, include.dirs = F)) / 
    #                            length(list.files(file.path(data_dir, '20200219'), recursive = T, include.dirs = F)))
    #                )
    #          )
    #)
    #
    #output$pct_anomaly <- renderText(
    #    print(paste('Pct anomaly (so far):', 
    #                    (
    #                        length(list.files(anomaly_path, recursive = T, include.dirs = F)) / 
    #                            (length(list.files(anomaly_path, recursive = T, include.dirs = F)) +
    #                                 length(list.files(normal_path, recursive = T, include.dirs = F)))
    #                    )
    #        )
    #    )
    #)
    #
    #output$pct_normal <- renderText(
    #    print(paste('Pct normal (so far):', 
    #                    (
    #                        length(list.files(normal_path, recursive = T, include.dirs = F)) / 
    #                            (length(list.files(anomaly_path, recursive = T, include.dirs = F)) +
    #                                 length(list.files(normal_path, recursive = T, include.dirs = F)))
    #                    )
    #    )
    #    )
    #)
    
    ######################################################################
    # Plots
    ######################################################################
    
    #---------------------------------------------------------------------
    # Respiration Analysis tabBox > OVerview
    
    # Overview with all variables
    output$plot_overview <- renderPlotly({
        
        # get preporcessed input
        raw_df <- readr::read_csv(file.path(wd, unclassified_path, file_list()[file_id()]), col_names = F)
        raw_df <- raw_df[,1:3]
        colnames(raw_df) <- c('MASK_PRESS_DATA', 'RESP_FLOW_DATA', 'DELIVERED_VOLUME_DATA') 

        
        raw_df <- raw_df %>% 
            mutate(
                step = seq.int(1, NROW(raw_df))
            ) %>% 
            gather(key ='MeasureType', value='value', 'MASK_PRESS_DATA', 'RESP_FLOW_DATA', 'DELIVERED_VOLUME_DATA') 

        # create plot
        p <-  ggplot(raw_df, aes(x = step, y = value)) +
            geom_line() +
            geom_point(size = .5,
                       aes(text = sprintf(
                           "Value: %s", value
                       ))) +

            facet_wrap( ~ MeasureType, ncol = 1, scales = 'free_y') +
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
    #output$raw_df = DT::renderDataTable({
    #    raw_df() %>% 
    #        dplyr::mutate(
    #            date = format(date, format = '%Y-%m-%d %H:%M:%OS4')
    #        ) %>% 
    #        tidyr::spread(
    #            key = category,
    #            value = value
    #        )
    #}, options = list(pageLength = 15))
    #
    ##---------------------------------------------------------------------
    ## Your classifications tabBox > Data table
    #
    ## shows the classification that the user made
    #output$classification_df <- renderDataTable({
    #    datatable(
    #        user_classification_df(), 
    #        options = list(dom = 'ft'))
    #})
})
