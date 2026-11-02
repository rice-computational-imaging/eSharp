"""CPU checks for Stage-I and calibrated Marigold evaluation behavior."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Marigold-ESharp'))
import src.trainer.marigold_trainer as trainer_module


class CheckpointSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'unet').mkdir()
        self.unet = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)
        torch.save(self.unet.state_dict(), self.root / 'unet/diffusion_pytorch_model.bin')

    def tearDown(self):
        self.temp.cleanup()

    def trainer(self, head_enabled):
        trainer = trainer_module.MarigoldTrainer.__new__(trainer_module.MarigoldTrainer)
        trainer.model = SimpleNamespace(unet=torch.nn.Linear(1, 1))
        trainer.calibration_head = torch.nn.Linear(1, 1) if head_enabled else None
        trainer.use_calibration_head = head_enabled
        trainer.device = torch.device('cpu')
        trainer.effective_iter = 0
        trainer.epoch = 1
        return trainer

    def load(self, trainer, require_head):
        trainer.load_checkpoint(str(self.root), load_trainer_state=False,
                                resume_lr_scheduler=False, require_calibration_head=require_head)

    def test_stage1_eval_and_stage2_initialization_allow_missing_head(self):
        self.load(self.trainer(False), True)
        self.load(self.trainer(True), False)

    def test_calibrated_eval_rejects_missing_head(self):
        with self.assertRaisesRegex(FileNotFoundError, 'requires a trained calibration head'):
            self.load(self.trainer(True), True)

    def test_head_file_and_trainer_state_fallback(self):
        trainer = self.trainer(True)
        head_file = self.root / 'calibration_head.ckpt'
        torch.save(self.head.state_dict(), head_file)
        self.load(trainer, True)
        torch.testing.assert_close(trainer.calibration_head.weight, self.head.weight)
        head_file.unlink()
        torch.save({'calibration_head': self.head.state_dict()}, self.root / 'trainer.ckpt')
        trainer = self.trainer(True)
        self.load(trainer, True)
        torch.testing.assert_close(trainer.calibration_head.weight, self.head.weight)

    def test_eval_only_can_skip_best_checkpoint_without_changing_training_default(self):
        for save_best in (False, True):
            trainer = self.trainer(False)
            dataset = SimpleNamespace(disp_name='example', filename_ls_path='split.txt')
            trainer.val_loaders = [SimpleNamespace(dataset=dataset)]
            trainer.val_metrics = object()
            trainer.validate_single_dataset = MagicMock(return_value={'rmse': 1.0})
            trainer._append_eval_metrics_csv = MagicMock()
            trainer.out_dir_eval = str(self.root)
            trainer.main_val_metric = 'rmse'
            trainer.main_val_metric_goal = 'minimize'
            trainer.best_metric = float('inf')
            trainer.save_checkpoint = MagicMock()
            with patch.object(trainer_module.tb_logger, 'log_dic'), patch.object(trainer_module, 'eval_dic_to_text', return_value='metrics'):
                if save_best:
                    trainer.validate()
                    trainer.save_checkpoint.assert_called_once()
                else:
                    trainer.validate(save_best_checkpoint=False)
                    trainer.save_checkpoint.assert_not_called()


if __name__ == '__main__':
    unittest.main()
