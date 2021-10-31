package lin

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dchkaa is the main test program for the DOUBLE PRECISION LAPACK
// linear equation routines
//
// The program must be driven by a short data file. The first 15 records
// (not including the first comment  line) specify problem dimensions
// and program options using list-directed input. The remaining lines
// specify the LAPACK test paths and the number of matrix types to use
// in testing.  An annotated example of a data file can be obtained by
// deleting the first 3 characters from the following 40 lines:
// Data file for testing DOUBLE PRECISION LAPACK linear eqn. routines
// 7                      Number of values of M
// 0 1 2 3 5 10 16        Values of M (row dimension)
// 7                      Number of values of N
// 0 1 2 3 5 10 16        Values of N (column dimension)
// 1                      Number of values of NRHS
// 2                      Values of NRHS (number of right hand sides)
// 5                      Number of values of NB
// 1 3 3 3 20             Values of NB (the blocksize)
// 1 0 5 9 1              Values of NX (crossover point)
// 3                      Number of values of RANK
// 30 50 90               Values of rank (as a % of N)
// 20.0                   Threshold value of test ratio
// T                      Put T to test the LAPACK routines
// T                      Put T to test the driver routines
// T                      Put T to test the error exits
// Dge   11               List types on next line if 0 < NTYPES < 11
// Dgb    8               List types on next line if 0 < NTYPES <  8
// Dgt   12               List types on next line if 0 < NTYPES < 12
// Dpo    9               List types on next line if 0 < NTYPES <  9
// Dps    9               List types on next line if 0 < NTYPES <  9
// Dpp    9               List types on next line if 0 < NTYPES <  9
// Dpb    8               List types on next line if 0 < NTYPES <  8
// Dpt   12               List types on next line if 0 < NTYPES < 12
// Dsy   10               List types on next line if 0 < NTYPES < 10
// Dsr   10               List types on next line if 0 < NTYPES < 10
// Dsk   10               List types on next line if 0 < NTYPES < 10
// Dsa   10               List types on next line if 0 < NTYPES < 10
// Ds2   10               List types on next line if 0 < NTYPES < 10
// Dsp   10               List types on next line if 0 < NTYPES < 10
// Dtr   18               List types on next line if 0 < NTYPES < 18
// Dtp   18               List types on next line if 0 < NTYPES < 18
// Dtb   17               List types on next line if 0 < NTYPES < 17
// Dqr    8               List types on next line if 0 < NTYPES <  8
// Drq    8               List types on next line if 0 < NTYPES <  8
// Dlq    8               List types on next line if 0 < NTYPES <  8
// Dql    8               List types on next line if 0 < NTYPES <  8
// Dqp    6               List types on next line if 0 < NTYPES <  6
// Dt2    3               List types on next line if 0 < NTYPES <  3
// Dls    6               List types on next line if 0 < NTYPES <  6
// Deq
// Dqt
// Dqx
// Dtq
// Dxq
// Dts
// Dhh
func TestDlin(t *testing.T) {
	var c2 string
	var eps, threq float64
	var i, j, kdmax, la, lafac, lda, nb, nmats, versMajor, versMinor, versPatch int

	nmax := 132
	maxrhs := 16
	kdmax = nmax + (nmax+1)/4
	dotype := make([]bool, 30)
	path := make([]byte, 3)
	e := vf(nmax)
	rwork := vf(5*nmax + 2*maxrhs)
	s := vf(2 * nmax)
	iwork := make([]int, 25*nmax)
	piv := make([]int, nmax)
	a := func() []*mat.Matrix {
		arr := make([]*mat.Matrix, 7)
		for u := 0; u < 7; u++ {
			arr[u] = mf(kdmax+1, nmax, opts)
		}
		return arr
	}()
	b := func() []*mat.Matrix {
		arr := make([]*mat.Matrix, 4)
		for u := 0; u < 4; u++ {
			arr[u] = mf(nmax, maxrhs, opts)
		}
		return arr
	}()
	work := mf(nmax, 3*nmax+maxrhs+30, opts)
	iparms := &gltest.Common.Claenv.Iparms
	*iparms = make([]int, 9)

	threq = 2.0

	lda = nmax

	//     Report values of parameters.
	versMajor, versMinor, versPatch = golapack.Ilaver()
	fmt.Printf(" Tests of the DOUBLE PRECISION LAPACK routines \n LAPACK VERSION %1d.%1d.%1d\n\n The following parameter values will be used:\n", versMajor, versMinor, versPatch)

	mval := []int{0, 1, 2, 3, 5, 10, 50}
	nm := len(mval)
	fmt.Printf("    %4s:", "m      ")
	for i = 1; i <= nm; i++ {
		fmt.Printf("  %6d", mval[i-1])
	}
	fmt.Printf("\n")

	nval := []int{0, 1, 2, 3, 5, 10, 50}
	nn := len(nval)
	fmt.Printf("    %4s:", "n      ")
	for i = 1; i <= nn; i++ {
		fmt.Printf("  %6d", nval[i-1])
	}
	fmt.Printf("\n")

	nsval := []int{1, 2, 15}
	nns := len(nsval)
	fmt.Printf("    %4s:", "nrhs   ")
	for i = 1; i <= nns; i++ {
		fmt.Printf("  %6d", nsval[i-1])
	}
	fmt.Printf("\n")

	nbval := []int{1, 3, 3, 3, 20}
	nnb := len(nbval)
	fmt.Printf("    %4s:", "nb     ")
	for i = 1; i <= nnb; i++ {
		fmt.Printf("  %6d", nbval[i-1])
	}
	fmt.Printf("\n")

	//     Set NBVAL2 to be the set of unique values of NB
	nbval2 := make([]int, nnb)
	nnb2 := 0
	for i = 1; i <= nnb; i++ {
		nb = nbval[i-1]
		for j = 1; j <= nnb2; j++ {
			if nb == nbval2[j-1] {
				goto label60
			}
		}
		nnb2 = nnb2 + 1
		nbval2[nnb2-1] = nb
	label60:
	}

	nxval := []int{1, 0, 5, 9, 1}
	fmt.Printf("    %4s:", "nx     ")
	for i = 1; i <= nnb; i++ {
		fmt.Printf("  %6d", nxval[i-1])
	}
	fmt.Printf("\n")

	rankval := []int{30, 50, 90}
	nrank := 3
	fmt.Print("rank % of n:")
	for i = 1; i <= nrank; i++ {
		fmt.Printf("  %6d", rankval[i-1])
	}
	fmt.Printf("\n")

	thresh := 30.0
	fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)

	tstchk := true
	tstdrv := true
	tsterr := true

	//     Calculate and print the machine dependent constants.
	eps = golapack.Dlamch(Underflow)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "underflow", eps)
	eps = golapack.Dlamch(Overflow)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "overflow ", eps)
	eps = golapack.Dlamch(Epsilon)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "precision", eps)

	nrhs := nsval[0]

	for _, c2 = range []string{"Dge", "Dgb", "Dgt", "Dpo", "Dps", "Dpp", "Dpb", "Dpt", "Dsy", "Dsr", "Dsk", "Dsa", "Ds2", "Dsp", "Dtr", "Dtp", "Dtb", "Dqr", "Drq", "Dlq", "Dql", "Dqp", "Dtz", "Dls", "Deq", "Dqt", "Dqx", "Dxq", "Dtq", "Dts", "Dhh"} {
		switch c2 {
		case "Dge":
			//        GE:  general matrices
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				dchkge(dotype, mval, nval, nbval2, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvge(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dgb":
			//        GB:  general banded matrices
			la = (2*kdmax + 1) * nmax
			lafac = (3*kdmax + 1) * nmax
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				dchkgb(dotype, mval, nval, nnb2, nbval2, nsval, thresh, tsterr, a[0].VectorIdx(0), la, a[2].VectorIdx(0), lafac, b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvgb(dotype, nn, nval, nrhs, thresh, tsterr, a[0].VectorIdx(0), la, a[2].VectorIdx(0), lafac, a[5].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dgt":
			//        GT:  general tridiagonal matrices
			nmats = 12
			alareq(nmats, &dotype)

			if tstchk {
				dchkgt(dotype, nval, nsval, thresh, tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvgt(dotype, nn, nval, nrhs, thresh, tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dpo":
			//        PO:  positive definite matrices
			nmats = 9
			alareq(nmats, &dotype)

			if tstchk {
				dchkpo(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvpo(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dps":
			//        PS:  positive semi-definite matrices
			nmats = 9

			alareq(nmats, &dotype)

			if tstchk {
				dchkps(dotype, nn, nval, nnb2, nbval2, nrank, rankval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), piv, work.VectorIdx(0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dpp":
			//        PP:  positive definite packed matrices
			nmats = 9
			alareq(nmats, &dotype)

			if tstchk {
				dchkpp(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvpp(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dpb":
			//        PB:  positive definite banded matrices
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				dchkpb(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvpb(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dpt":
			//        PT:  positive definite tridiagonal matrices
			nmats = 12
			alareq(nmats, &dotype)

			if tstchk {
				dchkpt(dotype, nn, nval, nns, nsval, thresh, tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvpt(dotype, nn, nval, nrhs, thresh, tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dsy":
			//        SY:  symmetric indefinite matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				dchksy(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvsy(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dsr":
			//        SR:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				dchksyRook(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvsyRook(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dsk":
			//        SK:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			//             differnet matrix storage format than SR path version.
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				dchksyRk(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), e, a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvsyRk(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), e, a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dsa":
			//        SA:  symmetric indefinite matrices,
			//             with partial (Aasen's) pivoting algorithm
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				dchksyAa(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvsyAa(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Ds2":
			//        SA:  symmetric indefinite matrices,
			//             with partial (Aasen's) pivoting algorithm
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				dchksyAa2stage(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvsyAa2stage(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dsp":
			//        SP:  symmetric indefinite packed matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				dchksp(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				ddrvsp(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Dtr":
			//        TR:  triangular matrices
			nmats = 18
			alareq(nmats, &dotype)

			if tstchk {
				dchktr(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dtp":
			//        TP:  triangular packed matrices
			nmats = 18
			alareq(nmats, &dotype)

			if tstchk {
				dchktp(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dtb":
			//        TB:  triangular banded matrices
			nmats = 17
			alareq(nmats, &dotype)

			if tstchk {
				dchktb(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dqr":
			//        QR:  QR factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				dchkqr(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dlq":
			//        LQ:  LQ factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				dchklq(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dql":
			//        QL:  QL factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				dchkql(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Drq":
			//        RQ:  RQ factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				dchkrq(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dqp":
			//        QP:  QR factorization with pivoting
			nmats = 6
			alareq(nmats, &dotype)

			if tstchk {
				dchkq3(dotype, nm, mval, nn, nval, nnb, nbval, nxval, thresh, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dtz":
			//        TZ:  Trapezoidal matrix
			nmats = 3
			alareq(nmats, &dotype)

			if tstchk {
				dchktz(dotype, nm, mval, nn, nval, thresh, tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dls":
			//        LS:  Least squares drivers
			nmats = 6
			alareq(nmats, &dotype)

			if tstdrv {
				ddrvls(dotype, nm, mval, nn, nval, nns, nsval, nnb, nbval, nxval, thresh, tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), rwork, rwork.Off(nmax), t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Deq":
			//        EQ:  Equilibration routines for general and positive definite
			//             matrices (THREQ should be between 2 and 10)
			if tstchk {
				dchkeq(threq, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dqt":
			//        QT:  QRT routines for general matrices
			if tstchk {
				dchkqrt(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dqx":
			//        QX:  QRT routines for triangular-pentagonal matrices
			if tstchk {
				dchkqrtp(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dtq":
			//        TQ:  LQT routines for general matrices
			if tstchk {
				dchklqt(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dxq":
			//        XQ:  LQT routines for triangular-pentagonal matrices
			if tstchk {
				dchklqtp(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dts":
			//        TS:  QR routines for tall-skinny matrices
			if tstchk {
				dchktsqr(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Dhh":
			//        HH:  Householder reconstruction for tall-skinny matrices
			if tstchk {
				dchkorhrCol(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		default:
			fmt.Printf("\n %3s:  Unrecognized path name\n", path)
		}
	}

	//     Branch to this line when the last record is read.

	fmt.Print("\n End of tests\n")
}

// Zchkaa is the main test program for the COMPLEX*16 linear equation
// routines.
//
// The program must be driven by a short data file. The first 15 records
// (not including the first comment  line) specify problem dimensions
// and program options using list-directed input. The remaining lines
// specify the LAPACK test paths and the number of matrix types to use
// in testing.  An annotated example of a data file can be obtained by
// deleting the first 3 characters from the following 42 lines:
// Data file for testing COMPLEX*16 LAPACK linear equation routines
// 7                      Number of values of M
// 0 1 2 3 5 10 16        Values of M (row dimension)
// 7                      Number of values of N
// 0 1 2 3 5 10 16        Values of N (column dimension)
// 1                      Number of values of NRHS
// 2                      Values of NRHS (number of right hand sides)
// 5                      Number of values of NB
// 1 3 3 3 20             Values of NB (the blocksize)
// 1 0 5 9 1              Values of NX (crossover point)
// 3                      Number of values of RANK
// 30 50 90               Values of rank (as a % of N)
// 30.0                   Threshold value of test ratio
// T                      Put T to test the LAPACK routines
// T                      Put T to test the driver routines
// T                      Put T to test the error exits
// Zge   11               List types on next line if 0 < NTYPES < 11
// Zgb    8               List types on next line if 0 < NTYPES <  8
// Zgt   12               List types on next line if 0 < NTYPES < 12
// Zpo    9               List types on next line if 0 < NTYPES <  9
// Zps    9               List types on next line if 0 < NTYPES <  9
// Zpp    9               List types on next line if 0 < NTYPES <  9
// Zpb    8               List types on next line if 0 < NTYPES <  8
// Zpt   12               List types on next line if 0 < NTYPES < 12
// Zhe   10               List types on next line if 0 < NTYPES < 10
// Zhr   10               List types on next line if 0 < NTYPES < 10
// Zhk   10               List types on next line if 0 < NTYPES < 10
// Zha   10               List types on next line if 0 < NTYPES < 10
// Zh2   10               List types on next line if 0 < NTYPES < 10
// Zsa   11               List types on next line if 0 < NTYPES < 10
// Zs2   11               List types on next line if 0 < NTYPES < 10
// Zhp   10               List types on next line if 0 < NTYPES < 10
// Zsy   11               List types on next line if 0 < NTYPES < 11
// Zsr   11               List types on next line if 0 < NTYPES < 11
// Zsk   11               List types on next line if 0 < NTYPES < 11
// Zsp   11               List types on next line if 0 < NTYPES < 11
// Ztr   18               List types on next line if 0 < NTYPES < 18
// Ztp   18               List types on next line if 0 < NTYPES < 18
// Ztb   17               List types on next line if 0 < NTYPES < 17
// Zqr    8               List types on next line if 0 < NTYPES <  8
// Zrq    8               List types on next line if 0 < NTYPES <  8
// ZLQ    8               List types on next line if 0 < NTYPES <  8
// Zql    8               List types on next line if 0 < NTYPES <  8
// Zqp    6               List types on next line if 0 < NTYPES <  6
// Ztz    3               List types on next line if 0 < NTYPES <  3
// Zls    6               List types on next line if 0 < NTYPES <  6
// ZEQ
// Zqt
// Zqx
// Zts
// Zhh
func TestZlin(t *testing.T) {
	var c2 string
	var tstchk, tstdrv, tsterr bool
	var eps, threq, thresh float64
	var i, j, kdmax, la, lafac, lda, maxrhs, nb, nm, nmats, nmax, nn, nnb, nnb2, nns, nrank, nrhs, versMajor, versMinor, versPatch int

	nmax = 132
	maxrhs = 16
	kdmax = nmax + (nmax+1)/4
	dotype := make([]bool, 30)
	path := make([]byte, 3)
	e := cvf(132)
	rwork := vf(150*nmax + 2*maxrhs)
	s := vf(2 * nmax)
	iwork := make([]int, 25*nmax)
	mval := make([]int, 12)
	nbval := make([]int, 12)
	nbval2 := make([]int, 12)
	nsval := make([]int, 12)
	nval := make([]int, 12)
	nxval := make([]int, 12)
	piv := make([]int, 132)
	rankval := make([]int, 12)
	a := func() []*mat.CMatrix {
		arr := make([]*mat.CMatrix, 7)
		for u := 0; u < 7; u++ {
			arr[u] = cmf(kdmax+1, nmax, opts)
		}
		return arr
	}()
	b := func() []*mat.CMatrix {
		arr := make([]*mat.CMatrix, 4)
		for u := 0; u < 4; u++ {
			arr[u] = cmf(nmax, maxrhs, opts)
		}
		return arr
	}()
	work := cmf(nmax, nmax+maxrhs+10, opts)

	iparms := &gltest.Common.Claenv.Iparms
	*iparms = make([]int, 9)

	threq = 2.0

	lda = nmax

	//     Report values of parameters.
	versMajor, versMinor, versPatch = golapack.Ilaver()
	fmt.Printf(" Tests of the COMPLEX*16 LAPACK routines \n LAPACK VERSION %1d.%1d.%1d\n\n The following parameter values will be used:\n", versMajor, versMinor, versPatch)

	mval = []int{0, 1, 2, 3, 5, 10, 50}
	nm = len(mval)
	fmt.Printf("    %4s:", "m      ")
	for i = 1; i <= nm; i++ {
		fmt.Printf("  %6d", mval[i-1])
	}
	fmt.Printf("\n")

	nval = []int{0, 1, 2, 3, 5, 10, 50}
	nn = len(nval)
	fmt.Printf("    %4s:", "n      ")
	for i = 1; i <= nn; i++ {
		fmt.Printf("  %6d", nval[i-1])
	}
	fmt.Printf("\n")

	nsval = []int{1, 2, 15}
	nns = len(nsval)
	fmt.Printf("    %4s:", "nrhs   ")
	for i = 1; i <= nns; i++ {
		fmt.Printf("  %6d", nsval[i-1])
	}
	fmt.Printf("\n")

	nbval = []int{1, 3, 3, 3, 20}
	nnb = len(nbval)
	fmt.Printf("    %4s:", "nb     ")
	for i = 1; i <= nnb; i++ {
		fmt.Printf("  %6d", nbval[i-1])
	}
	fmt.Printf("\n")

	//     Set NBVAL2 to be the set of unique values of NB
	nbval2 = make([]int, nnb)
	nnb2 = 0
	for i = 1; i <= nnb; i++ {
		nb = nbval[i-1]
		for j = 1; j <= nnb2; j++ {
			if nb == nbval2[j-1] {
				goto label60
			}
		}
		nnb2 = nnb2 + 1
		nbval2[nnb2-1] = nb
	label60:
	}

	nxval = []int{1, 0, 5, 9, 1}
	fmt.Printf("    %4s:", "nx     ")
	for i = 1; i <= nnb; i++ {
		fmt.Printf("  %6d", nxval[i-1])
	}
	fmt.Printf("\n")

	rankval = []int{30, 50, 90}
	nrank = 3
	fmt.Print("rank % of n:")
	for i = 1; i <= nrank; i++ {
		fmt.Printf("  %6d", rankval[i-1])
	}
	fmt.Printf("\n")

	thresh = 30.0
	fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)

	tstchk = true
	tstdrv = true
	tsterr = true

	//     Calculate and print the machine dependent constants.
	eps = golapack.Dlamch(Underflow)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "underflow", eps)
	eps = golapack.Dlamch(Overflow)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "overflow ", eps)
	eps = golapack.Dlamch(Epsilon)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "precision", eps)

	nrhs = nsval[0]

	//     Check first character for correct precision.
	for _, c2 = range []string{"Zge", "Zgb", "Zgt", "Zpo", "Zps", "Zpp", "Zpb", "Zpt", "Zhe", "Zhr", "Zhk", "Zha", "Zh2", "Zsa", "Zs2", "Zhp", "Zsy", "Zsr", "Zsk", "Zsp", "Ztr", "Ztp", "Ztb", "Zqr", "Zrq", "Zlq", "Zql", "Zqp", "Ztz", "Zls", "Zeq", "Zqt", "Zqx", "Zxq", "Ztq", "Zts", "Zhh"} {
		switch c2 {
		case "Zge":
			//        GE:  general matrices
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				zchkge(dotype, nm, mval, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvge(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVectorIdx(0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zgb":
			//        GB:  general banded matrices
			la = (2*kdmax + 1) * nmax
			lafac = (3*kdmax + 1) * nmax
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				zchkgb(dotype, nm, mval, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, a[0].CVector(0, 0), la, a[2].CVector(0, 0), lafac, b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvgb(dotype, nn, nval, nrhs, thresh, tsterr, a[0].CVector(0, 0), la, a[2].CVector(0, 0), lafac, a[5].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zgt":
			//        GT:  general tridiagonal matrices
			nmats = 12
			alareq(nmats, &dotype)

			if tstchk {
				zchkgt(dotype, nn, nval, nns, nsval, thresh, tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvgt(dotype, nn, nval, nrhs, thresh, tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zpo":
			//        PO:  positive definite matrices
			nmats = 9
			alareq(nmats, &dotype)

			if tstchk {
				zchkpo(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvpo(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zps":
			//        PS:  positive semi-definite matrices
			nmats = 9

			alareq(nmats, &dotype)

			if tstchk {
				zchkps(dotype, nn, nval, nnb2, nbval2, nrank, rankval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), piv, work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zpp":
			//        PP:  positive definite packed matrices
			nmats = 9
			alareq(nmats, &dotype)

			if tstchk {
				zchkpp(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvpp(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zpb":
			//        PB:  positive definite banded matrices
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				zchkpb(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvpb(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zpt":
			//        PT:  positive definite tridiagonal matrices
			nmats = 12
			alareq(nmats, &dotype)

			if tstchk {
				zchkpt(dotype, nn, nval, nns, nsval, thresh, tsterr, a[0].CVector(0, 0), s, a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvpt(dotype, nn, nval, nrhs, thresh, tsterr, a[0].CVector(0, 0), s, a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zhe":
			//        HE:  Hermitian indefinite matrices
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				zchkhe(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvhe(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zhr":
			//        HR:  Hermitian indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				zchkheRook(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvheRook(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zhk":
			//        HK:  Hermitian indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			//             different matrix storage format than HR path version.
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				zchkheRk(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvheRk(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zha":
			//        HA:  Hermitian matrices,
			//             Aasen Algorithm
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				zchkheAa(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvheAa(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zh2":
			//        H2:  Hermitian matrices,
			//             with partial (Aasen's) pivoting algorithm
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				zchkheAa2stage(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvheAa2stage(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zhp":
			//        HP:  Hermitian indefinite packed matrices
			nmats = 10
			alareq(nmats, &dotype)

			if tstchk {
				zchkhp(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvhp(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zsy":
			//        SY:  symmetric indefinite matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				zchksy(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvsy(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zsr":
			//        SR:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				zchksyRook(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvsyRook(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zsk":
			//        SK:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			//             different matrix storage format than SR path version.
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				zchksyRk(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvsyRk(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zsa":
			//        SA:  symmetric indefinite matrices with Aasen's algorithm,
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				zchksyAa(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvsyAa(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zs2":
			//        S2:  symmetric indefinite matrices with Aasen's algorithm
			//             2 stage
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				zchksyAa2stage(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvsyAa2stage(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Zsp":
			//        SP:  symmetric indefinite packed matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 11
			alareq(nmats, &dotype)

			if tstchk {
				zchksp(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				zdrvsp(dotype, nn, nval, nrhs, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "Ztr":
			//        TR:  triangular matrices
			nmats = 18
			alareq(nmats, &dotype)

			if tstchk {
				zchktr(dotype, nn, nval, nnb2, nbval2, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Ztp":
			//        TP:  triangular packed matrices
			nmats = 18
			alareq(nmats, &dotype)

			if tstchk {
				zchktp(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Ztb":
			//        TB:  triangular banded matrices
			nmats = 17
			alareq(nmats, &dotype)

			if tstchk {
				zchktb(dotype, nn, nval, nns, nsval, thresh, tsterr, lda, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zqr":
			//        QR:  QR factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				zchkqr(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zlq":
			//        LQ:  LQ factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				zchklq(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zql":
			//        QL:  QL factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				zchkql(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zrq":
			//        RQ:  RQ factorization
			nmats = 8
			alareq(nmats, &dotype)

			if tstchk {
				zchkrq(dotype, nm, mval, nn, nval, nnb, nbval, nxval, nrhs, thresh, tsterr, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zeq":
			//        EQ:  Equilibration routines for general and positive definite
			//             matrices (THREQ should be between 2 and 10)
			if tstchk {
				zchkeq(threq, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Ztz":
			//        TZ:  Trapezoidal matrix
			nmats = 3
			alareq(nmats, &dotype)

			if tstchk {
				zchktz(dotype, nm, mval, nn, nval, &thresh, &tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), s, b[0].CVector(0, 0), work.CVector(0, 0), rwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zqp":
			//        QP:  QR factorization with pivoting
			nmats = 6
			alareq(nmats, &dotype)

			if tstchk {
				zchkq3(dotype, nm, mval, nn, nval, nnb, nbval, nxval, thresh, a[0].CVector(0, 0), a[1].CVector(0, 0), s, b[0].CVector(0, 0), work.CVector(0, 0), rwork, iwork, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zls":
			//        LS:  Least squares drivers
			nmats = 6
			alareq(nmats, &dotype)

			if tstdrv {
				zdrvls(dotype, nm, mval, nn, nval, nns, nsval, nnb, nbval, nxval, thresh, tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), s, s.Off(nmax), t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zqt":
			//        QT:  QRT routines for general matrices
			if tstchk {
				zchkqrt(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zqx":
			//        QX:  QRT routines for triangular-pentagonal matrices
			if tstchk {
				zchkqrtp(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Ztq":
			//        TQ:  LQT routines for general matrices
			if tstchk {
				zchklqt(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zxq":
			//        XQ:  LQT routines for triangular-pentagonal matrices
			if tstchk {
				zchklqtp(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zts":
			//        TS:  QR routines for tall-skinny matrices
			if tstchk {
				zchktsqr(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "Zhh":
			//        HH:  Householder reconstruction for tall-skinny matrices
			if tstchk {
				zchkunhrCol(thresh, tsterr, nm, mval, nn, nval, nnb, nbval, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		default:
			t.Fail()
			fmt.Printf("\n %3s:  Unrecognized path name\n", path)
		}
	}

	//     Branch to this line when the last record is read.

	fmt.Printf("\n End of tests\n")
}

func TestMatrixInverse(t *testing.T) {
	var info int
	var err error

	s := [][][]complex128{
		// {
		// 	{1 + 0i, 1 - 2i},
		// 	{8 - 2i, 4 + 2i},
		// },
		{
			{2 - 6i, 4 + 1i, 2 + 0i},
			{10 + 6i, 2 - 8i, 1 + 0i},
			{2 - 2i, 4 + 1i, 4 - 6i},
		},
	}
	st := [][][]complex128{
		// {
		// 	{0.1 - 0.2i, 0.1 + 0.05i},
		// 	{0.1 + 0.4i, 0 - 0.05i},
		// },
		{
			{0.022688110281447446 + 0.08859850660539921i, 0.03360137851809305 - 0.01751866743251005i, 0.012349224583572665 - 0.0213957495692131i},
			{0.11783288846842586 + 0.010963949048890092i, 0.017028752914146704 + 0.06351995134642026i, -0.00957867351420752 - 0.03572997263236139i},
			{-0.01866743251005169 - 0.10137851809304999i, 0.016657093624353823 - 0.01723147616312464i, 0.05313038483630097 + 0.13469270534175762i},
		},
	}

	for k, val := range s {
		a := cmf(len(val), len(val[0]), mat.NewMatOptsCol())
		b := cmf(len(val), len(val[0]), mat.NewMatOpts())
		x := cmf(len(val), len(val[0]), mat.NewMatOptsCol())
		for i, r := range val {
			for j, c := range r {
				a.Set(i, j, c)
				b.Set(i, j, c)
				x.Set(i, j, st[k][i][j])
			}
		}

		lwork := a.Rows
		work := cvf(lwork)
		ipiv := make([]int, lwork)

		if info, err = golapack.Zgetrf(a.Rows, a.Cols, a, &ipiv); err != nil || info != 0 {
			panic(info)
		}
		if info, err = golapack.Zgetri(a.Cols, a, &ipiv, work, lwork); err != nil || info != 0 {
			panic(info)
		}
		for i := 0; i < a.Rows; i++ {
			for j := 0; j < a.Cols; j++ {
				if a.Get(i, j) != x.Get(i, j) {
					t.Errorf("Failed Col major inverse: got %v, want %v\n", a.Get(i, j), x.Get(i, j))
				}
			}
		}

		// if info, err = golapack.Zgetrf(b.Rows, b.Cols, b, &ipiv); err != nil || info != 0 {
		// 	panic(info)
		// }
		// if info, err = golapack.Zgetri(b.Cols, b, &ipiv, work, lwork); err != nil || info != 0 {
		// 	panic(info)
		// }
		// for i := 0; i < b.Rows; i++ {
		// 	for j := 0; j < b.Cols; j++ {
		// 		if b.Get(i, j) != x.Get(i, j) {
		// 			t.Errorf("Failed Row major inverse: got %v, want %v\n", b.Get(i, j), x.Get(i, j))
		// 		}
		// 	}
		// }
	}
}
