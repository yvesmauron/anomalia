library(rjson)
library(AzureAuth)
library(AzureRMR)
library(AzureGraph)
library(AzureStor)
library(readr)

# -----------------------------------------------------------------------------------
# General configuration for azure blob storage
# -----------------------------------------------------------------------------------

# get configuration file
config <- rjson::fromJSON(file = './config/adls.json')

# get end point of key
bl_endp_key <- AzureStor::storage_endpoint(Sys.getenv("END_POINT"), key = Sys.getenv("END_POINT_KEY"))

# get list of container
container_list <- AzureStor::list_storage_containers(bl_endp_key)

# get anomaly conatiner
anomaly_container <- get(config$anomaly_container, container_list)

# get labelling conatiner
label_container <- get(config$label_container, container_list)

# -----------------------------------------------------------------------------------
# Azure blob storage container wrapper function
# -----------------------------------------------------------------------------------

#' Get list of storage files
#'
#' @return
#' @export
#'
#' @examples
storage_file_list <- function() {
  
  # list files
  file_list <- AzureStor::list_storage_files(
    bl_endp_key, 
    container=anomaly_container
  )
  
  # return list
  return(file_list[[1]])
}



#' Read file form azure blob storage account
#'
#' @param file_name name to be read
#'
#' @return
#' @export
#'
#' @examples
storage_read_file <- function(file_name) {
  
  # create dir if not exists
  dir.create('./data', showWarnings = F, recursive = T)
  
  # download path
  download_path <- file.path('./data', file_name)
  
  # download file from 
  storage_download(
    container = anomaly_container,
    src = file_name,
    dest = download_path
  )
  
  # read file
  data_df <- readr::read_csv(
    download_path
  )
  
  # remove file
  file.remove(download_path)
  
  return(data_df)
}


#' Write files in a storage container
#'
#' @param data_df 
#'
#' @return
#' @export
#'
#' @examples
storage_write_file <- function(data_df) {
  
  # upload path
  current_time <- Sys.time()
  
  # add create_time to data_df
  data_df <- 
    data_df %>% 
    mutate(
      create_date = current_time
    )
  
  # populate file name and path
  file_name <- format(current_time, format='%Y%m%d_%H%M%S.csv')
  upload_path <- file.path('./data', file_name)
  
  # write file down
  write_csv(x = data_df, path = upload_path)
  
  # upload storage account
  storage_upload(
    container = label_container, 
    src = upload_path,
    dest = file_name
  )
  
  # remove file
  file.remove(upload_path)
}