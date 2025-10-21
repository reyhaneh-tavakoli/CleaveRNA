#!/usr/bin/perl
use strict;
use warnings;
use Getopt::Long;

# =====================================================
# prob.pl — Calculate per-base pairing probabilities
# and mark cleavage motifs
#
# Usage:
#   perl prob.pl -f input.fasta [-m GU] [-T 37] [-u 10]
# =====================================================

my ($input_file, $motif, $temp, $header, $u_len);
$motif  = "GU";  # default cleavage motif
$temp   = 37;    # default temperature
$u_len  = 10;    # default window length for lunp (U10)

GetOptions(
    "f=s" => \$input_file,
    "m=s" => \$motif,
    "T=f" => \$temp,
    "H=s" => \$header,
    "u=i" => \$u_len,
);

if (!$input_file) {
    die "Usage: perl $0 -f <input_fasta_file> [-m GU] [-T 37] [-u 10]\n";
}

# -----------------------------------------------------
# Run RNAfold (ViennaRNA)
# -----------------------------------------------------
print "Running RNAfold on $input_file (T=$temp)...\n";
system("RNAfold -p -T $temp < $input_file > RNAfold_output.txt") == 0
    or die "Error: RNAfold failed.\n";

# -----------------------------------------------------
# Read sequence (FASTA)
# -----------------------------------------------------
open(my $fh, '<', $input_file) or die "Cannot open $input_file: $!";
my $seq = '';
while(<$fh>) {
    next if /^>/;
    chomp;
    s/T/U/g;   # convert DNA to RNA (if present)
    $seq .= $_;
}
close $fh;
my @bases = split('', $seq);
my $len = length($seq);
print "Sequence length: $len bases\n";

# -----------------------------------------------------
# Find dot-plot (.dp.ps) file produced by RNAfold
# -----------------------------------------------------
my $dp_file;
for my $try ("${input_file}_dp.ps", "$input_file.dp.ps", "$input_file.ps", "dot.ps") {
    if (-e $try) { $dp_file = $try; last; }
}
unless ($dp_file) {
    (my $stem = $input_file) =~ s/\.[^.]+$//;
    $dp_file = "${stem}_dp.ps" if -e "${stem}_dp.ps";
}
die "Cannot find .dp.ps file for $input_file\n" unless $dp_file;
print "Using dot-plot file: $dp_file\n";

# -----------------------------------------------------
# Parse RNAfold's .dp.ps file for pairing probabilities
# Note: dp.ps stores sqrt(p) (by convention) -> square it
# -----------------------------------------------------
my @pp = ();    # 1-based indexing: pp[1] .. pp[$len]
open(my $dp, '<', $dp_file) or die "Cannot open $dp_file: $!";
while(<$dp>) {
    # match lines like:
    # 2 21 0.5501 ubox
    # or with 'boxed' instead of 'ubox'
    if (/^\s*(\d+)\s+(\d+)\s+([\d\.Ee+-]+)\s+(?:ubox|boxed)/) {
        my ($i, $j, $sqrtp) = ($1, $2, $3);
        # convert sqrt(p) -> p
        my $p = $sqrtp ** 2;
        $pp[$i] += $p;
        $pp[$j] += $p;
    }
}
close $dp;

# Ensure entries exist and cap at 1.0
for my $i (1..$len) {
    $pp[$i] ||= 0;
    $pp[$i] = 1 if $pp[$i] > 1;
    $pp[$i] = 0 if $pp[$i] < 0;
}

# -----------------------------------------------------
# Compute lunp (mean unpaired prob) over window length $u_len
# lunp at position i = mean_{k in window centered at i} (1 - pp[k])
# window length is u_len; we center it around i (integer floor)
# -----------------------------------------------------
my @lunp = ();
my $half = int($u_len / 2);
for my $i (1..$len) {
    my $sum = 0;
    my $count = 0;
    for (my $k = $i - $half; $k <= $i + $half; $k++) {
        next if $k < 1 || $k > $len;
        $sum += (1 - $pp[$k]);
        $count++;
    }
    # Avoid dividing by zero
    $lunp[$i] = $count ? ($sum / $count) : 0;
    # Clamp to [0,1]
    $lunp[$i] = 0 if $lunp[$i] < 0;
    $lunp[$i] = 1 if $lunp[$i] > 1;
}

# -----------------------------------------------------
# Detect cleavage motif positions (start positions)
# -----------------------------------------------------
my @cleav_positions;
while ($seq =~ /$motif/g) {
    push @cleav_positions, pos($seq) - length($&) + 1;
}
print "Found ", scalar(@cleav_positions), " cleavage sites for motif '$motif'\n";

# -----------------------------------------------------
# Write .prob output (tab-separated)
# Columns: Pos, pp, lunp, Cleav, seq
# -----------------------------------------------------
my $out_file = "$input_file.prob";
open(my $out, '>', $out_file) or die "Cannot open $out_file: $!";
print $out "Pos\tpp\tlunp\tCleav\tseq\n";
for my $i (1..$len) {
    my $pp_val   = sprintf("%.3f", $pp[$i]);
    my $lunp_val = sprintf("%.3f", $lunp[$i]);
    my $is_cleave = (grep { $_ == $i } @cleav_positions) ? 1 : 0;
    my $base     = $bases[$i-1] // '';
    print $out "$i\t$pp_val\t$lunp_val\t$is_cleave\t$base\n";
}
close $out;
print "Wrote: $out_file\n";

# -----------------------------------------------------
# Write .gle output for plotting (position vs pp)
# -----------------------------------------------------
my $gle_file = "$input_file.gle";
open(my $gle, '>', $gle_file) or die "Cannot open $gle_file: $!";
print $gle "begin graph\n";
if ($header) {
    (my $safe = $header) =~ s/"/'/g;  # replace double quotes with single quotes for safety
    print $gle "title \"$safe\"\n";
}
print $gle "xaxis 0 $len\n";
print $gle "yaxis 0 1\n";
print $gle "begin curve\n";
for my $i (1..$len) {
    printf $gle "%d %.6f\n", $i, $pp[$i];
}
print $gle "end curve\n";
print $gle "end graph\n";
close $gle;
print "Wrote: $gle_file\n";

print "✅ Done. Parsed pairing probabilities from $dp_file\n";
print " - pp: global pairing probability (from RNAfold, summed p from dotplot)\n";
print " - lunp: mean unpaired probability over window length $u_len (U$u_len)\n";

