import cv2
from pathlib import Path
import polars as pl
from typing import Callable, Optional
import logging
logger = logging.getLogger(__name__)


def get_image_paths(directory):
    image_files = []
    for ext in ['.jpg', '.png']:
        image_files.extend(Path(directory).glob(f"*{ext}"))
    return sorted([f for f in image_files])


def load_image(filepath, grayscale = False):
    if grayscale:
        return cv2.imread(str(filepath), flags=cv2.IMREAD_GRAYSCALE)
    return cv2.imread(str(filepath))


def init_dataframe(image_folder_path, label_file_path):
    images = get_image_paths(image_folder_path)
    label_df = pl.read_csv(label_file_path, has_header=True)
    image_df = pl.DataFrame({
        "sample_id": range(len(images)),
        "image": [path.stem for path in images],
        "image_path": [str(path) for path in images]
    })
    df = image_df.join(
        label_df,
        on="image",
        how="left",
    )
    return df


class Pipeline:
    def __init__(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        test_df: pl.DataFrame,
        output_path: Path = Path('processed')
    ):
        """Initialize pipeline with a polars DataFrame."""
        self.output_path = output_path
        self.splits = {
            "train": train_df,
            "valid": valid_df,
            "test": test_df,
        }

    @property
    def train_df(self):
        return self.splits["train"]
    
    @property
    def valid_df(self):
        return self.splits["valid"]
    
    @property
    def test_df(self):
        return self.splits["test"]
    
    def parse_image_path(self, image_name, suffix):
        sample_name = image_name.split("_")[1]
        return self.output_path / f"ISIC_{sample_name}_{suffix}.png"
    
    def save_processed_image(self, img, path):
        logger.debug(f'Saving image to {path}')
        if not cv2.imwrite(str(path), img):
            raise Exception("Faled to write image")
        return str(path)

    def apply_to_column(
        self,
        column: str,
        func: Callable,
        new_column: Optional[str] = None,
        on_split: list = ['test', 'valid'],
    ) -> 'Pipeline':
        """
        Apply a function to a specific column.
        
        Args:
            column: Name of the column to process
            func: Function to apply to the column
            new_column: Name of new column (if None, overwrites original)
            on_split: Dataframe split to apply on (if None, applied on test and valid)
        
        Returns:
            Updated Pipepline object
        """
        result_col = new_column if new_column else column
        for split in on_split:
            df = self.splits[split]
            processed = [func(element) for element in df[column]]
            self.splits[split] = df.with_columns(pl.Series(name=result_col, values=processed))

        return self
    
    def apply_to_image_and_save(
        self,
        column: str,
        func: Callable,
        new_column: str,
        on_split: list = ['test', 'valid'],
        skip_existing: bool = True,
    ) -> 'Pipeline':
        """
        Apply a function that returns an image, save image and create column of image paths.
        
        Args:
            column: Name of the column to process
            func: Function to apply to the column
            new_column: Name of new column
            on_split: Dataframe split to apply on (if None, applied on test and valid)
            skip_existing: If True, skip processing for images that already exist on disk
        
        Returns:
            Updated Pipepline object
        """
        total_processed = 0
        total_skipped = 0
        
        for split in on_split:
            df = self.splits[split]
            image_path_objs = [self.parse_image_path(Path(image_name).stem, new_column) for image_name in df[column]]
            
            processed_images = []
            skipped_count = 0
            processed_count = 0
            
            for idx, (element, image_path_obj) in enumerate(zip(df[column], image_path_objs)):
                if skip_existing and image_path_obj.exists():
                    logger.debug(f'Skipping existing image: {image_path_obj}')
                    processed_images.append(None)
                    skipped_count += 1
                else:
                    logger.debug(f'Processing image {idx+1}/{len(df[column])} for split {split}')
                    processed_images.append(func(element))
                    processed_count += 1

            for image, image_path_obj in zip(processed_images, image_path_objs):
                if image is not None:
                    self.save_processed_image(image, image_path_obj)

            logger.info(f'Split {split}: Processed {processed_count} images, skipped {skipped_count} existing images')

            total_processed += processed_count
            total_skipped += skipped_count

            image_paths_str = [str(p) for p in image_path_objs]
            self.splits[split] = df.with_columns(pl.Series(name=new_column, values=image_paths_str))

        total = total_processed + total_skipped
        print(f"Column '{new_column}' added: {total_processed} processed, {total_skipped} skipped ({total} total)")
        
        return self


    def print_df(self):
        print(f"Train DataFrame shape: {self.train_df.shape}")
        print(f"Columns: {self.train_df.columns}")
        print(self.train_df.head())
        print(f"Valid DataFrame shape: {self.valid_df.shape}")
        print(f"Columns: {self.valid_df.columns}")
        print(self.valid_df.head())
        print(f"Test DataFrame shape: {self.test_df.shape}")
        print(f"Columns: {self.test_df.columns}")
        print(self.test_df.head())

    def preview_processing(self, column: str, new_column: str, on_split: list = ['test'], max_show: int = 5):
        """
        Preview what files will be generated without actually processing.
        Useful for debugging skip_existing behavior.
        
        Args:
            column: Source column name
            new_column: Target column name for processed images
            on_split: Which splits to check
            max_show: Maximum number of examples to show per split
        
        Returns:
            Dictionary with statistics per split
        """
        results = {}
        
        for split in on_split:
            df = self.splits[split]
            image_path_objs = [self.parse_image_path(Path(image_name).stem, new_column) for image_name in df[column]]
            
            existing = [p for p in image_path_objs if p.exists()]
            missing = [p for p in image_path_objs if not p.exists()]
            
            results[split] = {
                'total': len(image_path_objs),
                'existing': len(existing),
                'missing': len(missing),
                'existing_files': existing[:max_show],
                'missing_files': missing[:max_show]
            }
            
            print(f"\nSplit: {split}")
            print(f"  Total images: {len(image_path_objs)}")
            print(f"  Already processed (will skip): {len(existing)}")
            print(f"  Need processing: {len(missing)}")
            
            if existing:
                print(f"  Example existing files (showing up to {max_show}):")
                for p in existing[:max_show]:
                    print(f"    ✓ {p.name}")
            
            if missing:
                print(f"  Example missing files (showing up to {max_show}):")
                for p in missing[:max_show]:
                    print(f"    ✗ {p.name}")
        
        return results

    def apply_to_columns(
        self,
        columns: list,
        func: Callable,
        new_column: str,
        on_split: list = ['test', 'valid'],
    ) -> 'Pipeline':
        """
        Apply a function that takes multiple column values as input.
        
        Args:
            columns: List of column names to pass to the function
            func: Function that takes multiple arguments (one per column)
            new_column: Name of new column for results
            on_split: Dataframe split to apply on
        
        Returns:
            Updated Pipeline object
        """
        for split in on_split:
            df = self.splits[split]
            
            column_values = [df[col] for col in columns]
            
            processed = [func(*row_values) for row_values in zip(*column_values)]
            
            self.splits[split] = df.with_columns(pl.Series(name=new_column, values=processed))
        
        print(f"Column '{new_column}' added from {len(columns)} input column(s)")
        
        return self
    
    def apply_to_columns_multi_output(
        self,
        columns: list,
        func: Callable,
        new_columns: list,
        on_split: list = ['test', 'valid'],
    ) -> 'Pipeline':
        """
        Apply a function that takes multiple inputs and returns multiple outputs.
        
        Args:
            columns: List of column names to pass to the function
            func: Function that returns a tuple/list of values
            new_columns: List of new column names for results
            on_split: Dataframe split to apply on
        
        Returns:
            Updated Pipeline object
        """
        for split in on_split:
            df = self.splits[split]
            
            column_values = [df[col] for col in columns]
            
            results = [func(*row_values) for row_values in zip(*column_values)]
            
            if results and isinstance(results[0], (tuple, list)):
                transposed = list(zip(*results))
            else:
                transposed = [results]
            
            for col_name, col_values in zip(new_columns, transposed):
                df = df.with_columns(pl.Series(name=col_name, values=list(col_values)))
            
            self.splits[split] = df
        
        print(f"{len(new_columns)} column(s) added: {', '.join(new_columns)}")
        
        return self
