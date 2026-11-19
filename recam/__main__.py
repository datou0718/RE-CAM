"""Command-line training and paper experiments."""
import argparse
from dataclasses import fields

from recam.experiments import METHODS, Settings, run
from recam.targets import FUNCTIONS


def main(argv=None):
    parser = argparse.ArgumentParser(description='RE-CAM waveform-based readout optimization')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('train', 'Train and evaluate one readout'),
                            ('benchmark', 'Run the paper readout or comparison-budget experiment')]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--output', required=True, help='Output directory; previous generated results are replaced')
        command.add_argument('--waveform', help='CSV in increasing similarity order; defaults to the bundled waveform')
        command.add_argument('--dataset', choices=['monte-carlo', 'distribution'], default='monte-carlo',
                             help='Evaluation dataset (default: monte-carlo)')
        if name == 'train':
            command.add_argument('--function', choices=FUNCTIONS, default='linear')
            command.add_argument('--method', choices=METHODS, default='recam')
        else:
            command.add_argument('--functions', nargs='+', choices=FUNCTIONS, default=list(FUNCTIONS))
            command.add_argument('--experiment', choices=['readout', 'comparison-budget'], default='readout',
                                 help='readout: Table II; comparison-budget: Table III')
        constraint_options = {
            'clock_ghz': 'System clock rate in GHz',
            'search_mhz': 'CAM search frequency in MHz',
            'latency_ps': 'Comparator latency in ps',
            'budget': 'Maximum comparisons per matchline per search',
        }
        nonideality_options = {
            'discharge_std': 'ML discharge speed variation (0.1 means 10%%)',
            'offset_mv': 'Comparator residual offset STD in mV',
            'noise_mv': 'Comparator sensing noise STD in mV',
        }
        constraints = command.add_argument_group('System Constraints')
        nonidealities = command.add_argument_group('Hardware Nonidealities')
        training = command.add_argument_group('Training and Evaluation')
        for field in fields(Settings):
            group = (constraints if field.name in constraint_options else
                     nonidealities if field.name in nonideality_options else training)
            description = constraint_options.get(field.name, nonideality_options.get(field.name, ''))
            options = dict(type=field.type, default=field.default,
                           help=f'{description} (default: {field.default})'.strip())
            if field.name == 'search':
                options['choices'] = ['auto', 'exhaustive']
                options['help'] = 'auto: exhaustive through 7 ticks, staged above 7; default: auto'
            group.add_argument('--'+field.name.replace('_', '-'), **options)
    args = parser.parse_args(argv)
    try:
        settings = Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)})
        single = args.command == 'train'
        run(args.output, settings, dataset=args.dataset,
            functions=[args.function] if single else args.functions,
            methods=[args.method] if single else None, waveform=args.waveform,
            experiment='readout' if single else args.experiment)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
