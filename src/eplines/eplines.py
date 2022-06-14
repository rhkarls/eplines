"""Heatmaps of empirical and exceedance probability of many (time-)series.

License: MIT (see LICENSE file)
Author: Reinert Huseby Karlsen, copyright 2022.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any

from matplotlib.axes import Axes

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.colors import Colormap

from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt


class ECDFLines:
    """Heatmaps of empirical and exceedance probability of many (time-)series.
    
    Parameters
    ----------
    mode : str, optional
        The mode determines how the ecdf is returned. 
        Alternatives are {‘ecdf’, ‘exceedance’, ‘deceedance’}. 
        See Notes below. The default is 'ecdf'.
    y_res : int, optional
        Resolution used on the y axis of the heatmap. The default is 100.
    x_res : int, optional
        Resolution used on the x axis of the heatmap. When set to None, the
        resolution is equal to the length of each y_line. The default is None.  

    Returns
    -------
    ECDFLines instance
    
    Notes
    -----
    
    TBD
    """

    def __init__(
            self,
            mode: Optional[str] = "ecdf",
            y_res: Optional[int] = 100,
            x_res: Optional[int] = None,
    ) -> None:

        # variables from arguments
        self.mode = mode
        self.y_res = y_res
        self.x_res = x_res

        # variables for fitting, data, plotting
        self.ecdfs = None
        self.xygrid = None
        self.x_plot = None
        self.y_plot = None
        self.x_plot_labels = None

    def ecdf(
            self,
            y_lines: Union[np.ndarray, pd.DataFrame],
            x: Optional[np.ndarray] = None,
    ):
        """
        Calculate the ecdf

        Parameters
        ----------
        y_lines : numpy.ndarray or pandas.DataFrame
            shape requirements for the nd.array.
        x : numpy.ndarray, optional
            what happens when None with y being array or dataframe
            shape requirement. The default is None.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # if user passed pandas dataframe, take the index as x and columns as ylines
        if hasattr(y_lines, "to_numpy"):
            try:
                if x is None:
                    x = y_lines.index.to_numpy()
                # transpose the dataframe to same format as numpy arrays
                y_lines = y_lines.T.to_numpy()
            except (AttributeError, TypeError):
                print(
                    "Please pass numpy arrays or pandas DataFrame to fit() method."
                )
                raise

        if x is None:
            x = np.arange(len(y_lines[0]))

        if self.x_res is None:
            self.x_res = len(x)

        # check x if numeric, so that grid and plots can be created from it
        # in any case, keep original x for plot labels
        self.x_plot_labels = x.copy()
        if not np.issubdtype(x.dtype, np.number):
            x = np.arange(len(x))

        self.x_plot = x
        self.y_plot = y_lines

        # Setup grid
        # noinspection PyArgumentList
        x_min = x.min()
        # noinspection PyArgumentList
        x_max = x.max()
        y_min = y_lines.min()
        y_max = y_lines.max()

        grid_size_x = self.x_res
        grid_size_y = self.y_res
        # Generate grid
        xygrid = np.mgrid[
                 x_min: x_max: complex(grid_size_x),
                 y_min: y_max: complex(grid_size_y),
                 ]

        self.xygrid = xygrid

        grid_x = xygrid[0]
        grid_y = xygrid[1]

        ecdfs = self._grid_ecdfs(x, y_lines, grid_x, grid_y)

        self.ecdfs = ecdfs

        return self

    def _grid_ecdfs(self, x, y_lines, grid_x, grid_y):

        ecdfs = np.empty((grid_x.shape[0], grid_y.shape[1]))

        for i in range(len(x)):
            value, ecdf = self._calc_ecdf(y_lines[:, i], weibull_plotting=True)

            # interpolate to the grid_y coordinates and fill edges
            ecdf_grid = np.interp(grid_y[0], value, ecdf, left=0, right=1)
            ecdfs[i] = ecdf_grid

        if self.mode == 'exceedance':
            ecdfs = 1 - ecdfs

        ecdfs = np.fliplr(ecdfs)

        return ecdfs

    def _calc_ecdf(self, x, weibull_plotting=False):
        """ Calculate the ecdf """

        # drop na values
        x = x[~np.isnan(x)]

        # Adjust plotting position
        if weibull_plotting:
            pp = 1
        else:
            pp = 0

        # use np.unique to count values. Returns sorted values
        value, freq = np.unique(x, return_counts=True)
        epdf = freq / (np.sum(freq) + pp)  # for Weibull plotting position (freq/(np.sum(freq)+1)
        ecdf = np.cumsum(epdf)

        return value, ecdf

    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            cmap: Optional[Union[str, Colormap]] = "viridis",
            aspect: Optional[Union[str, float]] = "auto",
            show_data: Optional[bool] = False,
            time_label_format: Optional[str] = "%Y-%m-%d",
            mask_to_data=False,
            mask_above=1,
            mask_below=0,
            **kwargs,
    ) -> tuple[Axes | Any, Any]:
        """
        Plot the ecdfs as a heatmap.

        Parameters
        ----------
        ax : matplotlib axes object, optional
            Axes to plot the heatmap on. If None a new figure and axes
            is created. The default is None.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap used for density heatmap.
            See https://matplotlib.org/stable/tutorials/colors/colormaps.html
            The default is 'viridis'.
        aspect : str or float, optional
            Aspect ration of the axes, see matplotlib.pyplot.imshow.
            The default is 'auto'.
        show_data : bool, optional
            Plot the data that the kde is based on.
            Plotted as black thin lines. The default is False.
        time_label_format : str, optional
            If x values are datetime/timestamps format to this string
            on the plot labels. See datetime.strftime
            The default is '%Y-%m-%d'.
        mask_to_data: bool, optional
            Mask the plotted ecdf to be limited to the range of the data.
        mask_above: float, optional
            Values above mask_above will not be plotted.
        mask_below: float, optional
            Values below mask_below will not be plotted.
        **kwargs : dict
            Additional keyword arguments for matplotlib.pyplot.imshow.

        Returns
        -------
        ax : matplotlib axes object
            The axes the heatmap is plotted on.
        ims : matplotlib imshow object

        """
        if self.ecdfs is None:
            raise NotCalculatedError("Nothing to plot. ")

        if ax is None:
            fig, ax = plt.subplots()

        ecdfs_plt = self.ecdfs.copy()

        if mask_to_data:
            ymins = self.y_plot.min(axis=0)
            ymaxs = self.y_plot.max(axis=0)

            # Take grid size - index of value since grid is flipped on y axis
            grid_y_size = len(self.xygrid[1][0])
            grid_y_min = grid_y_size - np.searchsorted(self.xygrid[1][0], ymins)
            grid_y_max = grid_y_size - np.searchsorted(self.xygrid[1][0], ymaxs)

            # Since grid is flipped its index min: and :max
            for col in np.arange(ecdfs_plt.shape[0]):
                ecdfs_plt[col, grid_y_min[col]:] = np.nan
                ecdfs_plt[col, :grid_y_max[col]] = np.nan

        if mask_above is not None:
            ecdfs_plt[self.ecdfs > mask_above] = np.nan

        if mask_below is not None:
            ecdfs_plt[self.ecdfs < mask_below] = np.nan

        ims = ax.imshow(
                ecdfs_plt.T,
                cmap=cmap,
                extent=[
                    np.nanmin(self.x_plot),
                    np.nanmax(self.x_plot),
                    np.nanmin(self.y_plot),
                    np.nanmax(self.y_plot),
                ],
                aspect=aspect,
                **kwargs,
            )

        if show_data:
            ax.plot(self.x_plot, self.y_plot.T, color="k", alpha=0.2, lw=0.1)

        # set x tick labels when x data is not numeric
        if not np.issubdtype(self.x_plot_labels.dtype, np.number):
            x_tick_locs = ax.get_xticks()
            x_tick_locs = x_tick_locs.astype(int)
            try:
                x_tick_labels = self.x_plot_labels[x_tick_locs]
            except IndexError:  # often the returned ticks are just outside of the range
                x_tick_locs = x_tick_locs[:-1]
                x_tick_labels = self.x_plot_labels[x_tick_locs]

            # use the time_label_format if datetime ticks
            # casting to datetime using item(), but needs to be as [s] then
            if np.issubdtype(x_tick_labels.dtype, np.datetime64):
                new_x_tick_labels = []
                for i, l in enumerate(x_tick_labels):
                    new_x_tick_labels.append(
                        l.astype("datetime64[s]")
                            .item()
                            .strftime(time_label_format)
                    )
            else:
                new_x_tick_labels = x_tick_labels

            ax.set_xticks(x_tick_locs)
            ax.set_xticklabels(new_x_tick_labels)

        return ax, ims


class NotCalculatedError(Exception):
    """Exception for not calculated error.

    Attributes
    ----------
    message : str
        error message
    """

    def __init__(self, message=""):
        self.message = message

    def __str__(self):
        return (
            f"{self.message}This ECDFLines instance does not yet have the ecdfs calculated. "
            f"Call ecdf() before using this method."
        )
