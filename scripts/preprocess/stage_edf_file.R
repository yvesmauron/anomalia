# load libraries
require(edf)
require(foreach)
require(doSNOW)
require(parallel)
require(progress)
#setwd("C:/Users/yvesm/OneDrive - Trivadis AG/projects/atemreich/atemteurer")
# --------------------------------------------------------------------------
# Set up data directory structure
# --------------------------------------------------------------------------
device <- 'resmed'
time_dir <- format(Sys.Date(), '%Y%m%d')
target_path <- file.path('./data', device, 'staging', time_dir)
unlink(target_path, recursive = T, force = T)
dir.create(target_path, showWarnings = F, recursive = T)

# --------------------------------------------------------------------------
# List stations
# --------------------------------------------------------------------------
stations <- list.dirs(file.path(getwd(), 'data', device, 'raw'), full.names = F, recursive = F)

for (station in stations) {
  print(paste('Processing station:', station))
  # --------------------------------------------------------------------------
  # List the files to extract
  # --------------------------------------------------------------------------
  edf_dir = file.path(getwd(), 'data', device, 'raw', station)
  edf_file_paths <- list.files(edf_dir, recursive = T, pattern = '*.edf')
  
  # --------------------------------------------------------------------------
  # Set up cluster for parallel execution (incl. progress bar)
  # --------------------------------------------------------------------------
  # create progress bar
  pb <- progress_bar$new(
    format = "  [:bar] :percent eta: :eta",
    total = length(edf_file_paths),
    clear = F,
    force = T
  )
  prop_progress <- function() { pb$tick() }
  
  # set num_cores if it is not set by the user
  num_cores <- detectCores()
  num_cores_ignore <- 1
  
  # create local cluster
  cl <- makeCluster(num_cores - num_cores_ignore, type = "SOCK")
  registerDoSNOW(cl)
  clusterEvalQ(cl, { 
    library(edf)
  })
  
  # --------------------------------------------------------------------------
  # read and extract information of edf files
  # --------------------------------------------------------------------------
  result <- foreach (file_name = edf_file_paths, .options.snow = list(progress=prop_progress)) %dopar% {
    
    # convert from edf format to list
    edf_file <- edf::read.edf(filename = file.path(edf_dir, file_name), read.annotations = F)
    
    # get global header information
    header_global <- t(unlist(edf_file$header.global))
    
    #record signal source file information
    header_global <- cbind(header_global, file_name, file_name)
    
    # get datetime of signal
    dt_signal <- strptime(paste(header_global[1,4], ' ', header_global[1,5], sep = ''), '%d.%m.%y %H.%M.%S')
    
    # get signal header information
    header_signal <- t(sapply(edf_file$header.signal, function(x) {
      unlist(x)
    }))
    
    #record signal source file information
    header_signal <- cbind(header_signal, file_name, file_name)
    
    # get the time the signales were recoded
    signal_time <- sapply(edf_file$signal, function(x) {
      x$t
    })
    
    # rename columns to distiguish between time and data later
    colnames(signal_time) <- paste(colnames(signal_time), 'T', sep = '_') 
    
    # get signal data 
    signal_data <- sapply(edf_file$signal, function(x) {
      x$data
    })
    
    signal_ts = format(signal_time[,1] + dt_signal, '%Y-%m-%d %H:%M:%OS4')
    
    # rename to signal data 
    colnames(signal_data) <- paste(colnames(signal_data), 'DATA', sep = '_')
    
    # combine signal time and signal data and record source file information
    signal <- cbind(TimeStamp = signal_ts, signal_time, signal_data, file_name, StationName=station)

    # signal
    write.csv(
      x = signal, 
      file = file.path(target_path, paste(file_name,  'csv', sep = '.')), 
      row.names = F
    )
    list (
      file = file_name,
      processing_time = Sys.time()
    )
  }
  # close cluster
  stopCluster(cl)
}
