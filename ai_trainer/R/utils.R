library(dplyr)


#' General function to preprocess file
#'
#' @param file_name 
#'
#' @return
#' @export
#'
#' @examples
read_raw_data <- function(file_name) {
  
  # download file from azure storage acccount
  raw_df <- storage_read_file(
    file_name = file_name
  )
  
  # return pp_df
  return(raw_df)
}


#' General function to preprocess file
#'
#' @param file_name 
#'
#' @return
#' @export
#'
#' @examples
preprocess_data <- function(raw_df) {
  
  # preprocess file 
  pp_df <- 
    raw_df %>% 
    as_tibble() %>%
    mutate(
      date = as.POSIXct(date, tz = 'CET', format = '%Y-%m-%d %H:%M:%OS'),
      category = as.factor(category)
    ) %>%
    mutate(
      date_numeric = as.numeric(date),
      date_string = format(date, format = '%Y-%m-%d %H:%M:%OS4')
    )
  
  # return pp_df
  return(pp_df)
}


window_time_frame_limits <- function(df) {
  return(
    c(
      min(df$date),
      max(df$date)
    )
  )
} 


remove_exisiting_rows <- function(df, id) {
  
  existing_id <- which(df$uid == id)

  if (length(existing_id) > 0) {
    df <- df[-existing_id, ]
  }
  
  return(df)
}


update_classification_df <- function(class_df, raw_df, u_id, is_correct = T) {
  
  # remove existing rows
  class_df <- remove_exisiting_rows(df = class_df, id = u_id)
  
  # get window limits
  window_limits <- window_time_frame_limits(raw_df)
  
  # rbind
  updated_df = rbind(
    data_frame(
      uid = u_id,
      station = 'tbd',
      from = format(window_limits[1], format = '%Y-%m-%d %H:%M:%OS4'),
      to = format(window_limits[2], format = '%Y-%m-%d %H:%M:%OS4'),
      is_anomaly = as.numeric(is_correct),
      class = 'tbd'
    ),
    class_df
  )  
  
  return(updated_df)
}

