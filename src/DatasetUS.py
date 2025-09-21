import torch
import torchvision
import numpy as np
import xarray as xr


class UpscaleDataset(torch.utils.data.Dataset):
    """
    Dataset class of images with a low resolution and a high resolution counterpart
    over the US continent.
    """

    def __init__(self, data_dir,
                 in_shape=(16, 32), out_shape=(128, 256),
                 year_start=1950, year_end=2001,
                 normalize_rawdata_mean=torch.Tensor([2.8504e+02]),
                 normalize_rawdata_std=torch.Tensor([12.7438]),
                 normalize_residual_mean=torch.Tensor([-9.4627e-05]),
                 normalize_residual_std=torch.Tensor([1.6042]),
                 constant_variables=None,
                 constant_variables_filename="ERA5_const_sfc_variables.nc"
                 ):
        """
        :param data_dir: path to the dataset directory
        :param in_shape: shape of the low resolution images
        :param out_shape: shape of the high resolution images
        :param year_start: starting year of file named samples_{year_start}.nc
        :param year_end: ending year of file named samples_{year_end}.nc
        :param normalize_mean: channel-wise mean values estimated over all samples
        for normalizing file
        :param normalize_std: channel-wise standard deviation values estimated
        over all samples for normalizing file
        """

        print("Opening files")
        self.filenames = [f"samples_{year}.nc" for year in range(year_start, year_end)]

        # Open first file for saving dimension info
        filename0 = self.filenames[0]
        path_to_file = data_dir + filename0
        ds = xr.open_dataset(path_to_file, engine="netcdf4")

        # Dimensions: lon, lat (global domain)
        self.lon_glob = ds.longitude
        self.lat_glob = ds.latitude
        self.varnames = ["temp"]
        self.n_var = len(self.varnames)


        # Select domain with size 128 x 256 (W x H)
        ds_US = ds.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
                       longitude=slice(233.6, 297.5))  # longitude ordered E to W
        self.lon = ds_US.longitude.values  # Convert to numpy for pickling
        self.lat = ds_US.latitude.values   # Convert to numpy for pickling
        self.nlon = self.W = len(self.lon)  # Width
        self.nlat = self.H = len(self.lat)  # Height

        # Concatenate other files
        for filename in self.filenames[1:]:
            path_to_file = data_dir + filename
            ds = xr.open_dataset(path_to_file, engine="netcdf4")
            # Select domain with size 256 x 256 (W x H)
            ds_US = xr.concat((ds_US,
                               ds.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
                                      longitude=slice(233.6, 297.5))),  # longitude ordered E to W
                              dim="time")

        print("All files accessed. Creating tensors")
        # WEEKLY DATA SUBSAMPLING - Update ntime to reflect weekly data
        # Original code (commented out):
        # self.ntime = len(ds_US.time)
        
        # New code: Calculate weekly data size
        self.ntime = len(ds_US.time) // 7  # Weekly data: every 7th day
        print(f"Original time steps: {len(ds_US.time)}")
        print(f"Weekly time steps: {self.ntime} (86% reduction)")

        # Convert xarray dataarrays into torch Tensor (loads into memory)
        t = torch.from_numpy(ds_US.VAR_2T.to_numpy()).float()

        # WEEKLY DATA SUBSAMPLING - Take every 7th day for 86% data reduction
        # Original code (commented out):
        # fine = torch.stack((t,), dim=1)
        
        # New code: Weekly sampling (every 7th day)
        print(f"Original data shape: {t.shape}")
        t_weekly = t[::7]  # Take every 7th day (weekly data)
        print(f"Weekly data shape: {t_weekly.shape} (86% reduction)")
        fine = torch.stack((t_weekly,), dim=1)

        # Transforms
        # Coarsen
        coarsen_transform = torchvision.transforms.Resize(in_shape,
                                                          interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                          antialias=True)
        interp_transform = torchvision.transforms.Resize(out_shape,
                                                         interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                         antialias=True)
        # Coarsen fine into coarse image, but interp to keep it on same grid
        # This will be our input into NN
        coarse = interp_transform(coarsen_transform(fine))
        # Calculate residual = fine - coarse. this will be our target
        residual = fine - coarse

        # Save unnormalized coarse and fine images for plotting
        self.coarse = coarse
        self.fine = fine

        # Normalize : use raw data means for coarse image
        normalize_rawdata_transform = torchvision.transforms.Normalize(normalize_rawdata_mean, normalize_rawdata_std)
        coarse_norm = normalize_rawdata_transform(coarse)

        # use residual means for the difference between them
        normalize_residual_transform = torchvision.transforms.Normalize(normalize_residual_mean, normalize_residual_std)
        residual_norm = normalize_residual_transform(residual)

        print(normalize_residual_std.shape)
        # Store normalization parameters for inverse normalization (convert to numpy for pickling)
        self.normalize_residual_std = normalize_residual_std.numpy()
        self.normalize_residual_mean = normalize_residual_mean.numpy()

        # Save
        self.targets = residual_norm     # targets  = normalized residual
        self.inputs = coarse_norm        # inputs   = normalized coarse

        # Define limits for plotting (plus/minus 2 sigma
        self.vmin = normalize_rawdata_mean - 2 * normalize_rawdata_std
        self.vmax = normalize_rawdata_mean + 2 * normalize_rawdata_std

        print(self.vmin, self.vmax)



        # Additional channels for constant variables
        self.constant_variables = constant_variables
        if constant_variables is not None:
            print("Opening constant variables file (e.g. land-sea mask, topography)")
            # Open file
            ds_const = xr.open_dataset(data_dir + constant_variables_filename,
                                       engine="netcdf4")
            ds_const = ds_const.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
                                    longitude=slice(233.6, 297.5))

            # Get torch tensors and concatenate
            self.const_var = torch.zeros((self.ntime,
                                          len(constant_variables),
                                          self.nlat,
                                          self.nlon),
                                         dtype=torch.float)

            for i, const_varname in enumerate(constant_variables):
                const_var = ds_const[const_varname]
                # normalize?
                if const_varname != "lsm":
                    print(f"Normalize {const_varname}")
                    # Use xarray for calculation but convert to numpy immediately
                    weighted_var = const_var.weighted(np.cos(np.radians(ds_const.latitude)))
                    mean_var = weighted_var.mean().values  # Convert to numpy immediately
                    std_var = weighted_var.std().values    # Convert to numpy immediately
                    print(f"Mean:{mean_var}, Std{std_var}")
                    const_var_norm = (const_var - mean_var) / std_var
                    const_var_tensor = torch.from_numpy(const_var_norm.to_numpy()).float()
                else:
                    # For lsm, just convert to numpy
                    const_var_tensor = torch.from_numpy(const_var.to_numpy()).float()
                
                # Replicate constant variable for all time steps
                self.const_var[:, i, :, :] = const_var_tensor.unsqueeze(0).expand(self.ntime, -1, -1)
            self.inputs = torch.concatenate((self.inputs, self.const_var), dim=1)

        # Dimensions from orig to coarse
        lat_coarse_inds = np.arange(0, len(self.lat), 8, dtype=int)
        lon_coarse_inds = np.arange(0, len(self.lon), 8, dtype=int)

        self.lon_coarse = self.lon[lon_coarse_inds]  # Convert to numpy indexing
        self.lat_coarse = self.lat[lat_coarse_inds]  # Convert to numpy indexing

        # Time embeddings - convert to numpy for pickling
        time_dt = ds_US.time.dt        # in datetime format
        
        # WEEKLY DATA SUBSAMPLING - Apply same sampling to time embeddings
        # Original code (commented out):
        # self.year = time_dt.year.values
        # self.month = time_dt.month.values
        # self.day = time_dt.day.values
        # self.hour = time_dt.hour.values
        
        # New code: Weekly sampling for time embeddings
        self.year = time_dt.year.values[::7]      # Every 7th day
        self.month = time_dt.month.values[::7]    # Every 7th day
        self.day = time_dt.day.values[::7]        # Every 7th day
        self.hour = time_dt.hour.values[::7]      # Every 7th day
        # day of year (1 to 360)
        self.doy = ((self.month - 1.) * 30 + (self.day - 1.))

        # Normalize and convert to numpy (load into mem)
        self.year_norm = (self.year - 1940.)/100
        self.doy_norm = self.doy/360.
        self.hour_norm = self.hour/24.

        # Torch arrays and float
        self.year_norm = torch.from_numpy(self.year_norm).float()
        self.doy_norm = torch.from_numpy(self.doy_norm).float()
        self.hour_norm = torch.from_numpy(self.hour_norm).float()

        print("Dataset initialized.")

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.inputs.shape[0]

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        return {"inputs": self.inputs[index],
                "targets": self.targets[index],
                "fine": self.fine[index],
                "coarse": self.coarse[index],
                "year": self.year_norm[index],
                "doy": self.doy_norm[index],
                "hour": self.hour_norm[index]}

    def inverse_normalize_residual(self, residual_norm):
        """Inverse normalize residual data"""
        std_tensor = torch.from_numpy(self.normalize_residual_std).to(residual_norm.device)
        mean_tensor = torch.from_numpy(self.normalize_residual_mean).to(residual_norm.device)
        return ((residual_norm * std_tensor[:, np.newaxis, np.newaxis]) +
                mean_tensor[:, np.newaxis, np.newaxis])
    
    def residual_to_fine_image(self, residual, coarse_image):
        return coarse_image + self.inverse_normalize_residual(residual)
    
    def plot_batch(self, coarse_image, fine_image, fine_image_pred, N=4):
        """Plot a batch of images for visualization"""
        try:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            # Fallback to simple matplotlib if cartopy not available
            import matplotlib.pyplot as plt
            
        fig, axes = plt.subplots(3, N, figsize=(4*N, 12))
        if N == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(N):
            # Coarse image
            ax = axes[0, i]
            if hasattr(self, 'lon') and hasattr(self, 'lat'):
                try:
                    # Try to use cartopy for proper projection
                    ax = plt.subplot(3, N, i+1, projection=ccrs.PlateCarree())
                    im = ax.pcolormesh(self.lon, self.lat, coarse_image[i, 0].numpy(), 
                                     transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS)
                    ax.set_title(f'Coarse {i+1}')
                except:
                    # Fallback to simple plot
                    ax.imshow(coarse_image[i, 0].numpy(), cmap='RdBu_r')
                    ax.set_title(f'Coarse {i+1}')
            else:
                ax.imshow(coarse_image[i, 0].numpy(), cmap='RdBu_r')
                ax.set_title(f'Coarse {i+1}')
            
            # Fine image (truth)
            ax = axes[1, i]
            if hasattr(self, 'lon') and hasattr(self, 'lat'):
                try:
                    ax = plt.subplot(3, N, N+i+1, projection=ccrs.PlateCarree())
                    im = ax.pcolormesh(self.lon, self.lat, fine_image[i, 0].numpy(), 
                                     transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS)
                    ax.set_title(f'Truth {i+1}')
                except:
                    ax.imshow(fine_image[i, 0].numpy(), cmap='RdBu_r')
                    ax.set_title(f'Truth {i+1}')
            else:
                ax.imshow(fine_image[i, 0].numpy(), cmap='RdBu_r')
                ax.set_title(f'Truth {i+1}')
            
            # Predicted image
            ax = axes[2, i]
            if hasattr(self, 'lon') and hasattr(self, 'lat'):
                try:
                    ax = plt.subplot(3, N, 2*N+i+1, projection=ccrs.PlateCarree())
                    im = ax.pcolormesh(self.lon, self.lat, fine_image_pred[i, 0].numpy(), 
                                     transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS)
                    ax.set_title(f'Predicted {i+1}')
                except:
                    ax.imshow(fine_image_pred[i, 0].numpy(), cmap='RdBu_r')
                    ax.set_title(f'Predicted {i+1}')
            else:
                ax.imshow(fine_image_pred[i, 0].numpy(), cmap='RdBu_r')
                ax.set_title(f'Predicted {i+1}')
        
        plt.tight_layout()
        return fig, axes

