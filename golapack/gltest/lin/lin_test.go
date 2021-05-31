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
// DGE   11               List types on next line if 0 < NTYPES < 11
// DGB    8               List types on next line if 0 < NTYPES <  8
// DGT   12               List types on next line if 0 < NTYPES < 12
// DPO    9               List types on next line if 0 < NTYPES <  9
// DPS    9               List types on next line if 0 < NTYPES <  9
// DPP    9               List types on next line if 0 < NTYPES <  9
// DPB    8               List types on next line if 0 < NTYPES <  8
// DPT   12               List types on next line if 0 < NTYPES < 12
// DSY   10               List types on next line if 0 < NTYPES < 10
// DSR   10               List types on next line if 0 < NTYPES < 10
// DSK   10               List types on next line if 0 < NTYPES < 10
// DSA   10               List types on next line if 0 < NTYPES < 10
// DS2   10               List types on next line if 0 < NTYPES < 10
// DSP   10               List types on next line if 0 < NTYPES < 10
// DTR   18               List types on next line if 0 < NTYPES < 18
// DTP   18               List types on next line if 0 < NTYPES < 18
// DTB   17               List types on next line if 0 < NTYPES < 17
// DQR    8               List types on next line if 0 < NTYPES <  8
// DRQ    8               List types on next line if 0 < NTYPES <  8
// DLQ    8               List types on next line if 0 < NTYPES <  8
// DQL    8               List types on next line if 0 < NTYPES <  8
// DQP    6               List types on next line if 0 < NTYPES <  6
// DTZ    3               List types on next line if 0 < NTYPES <  3
// DLS    6               List types on next line if 0 < NTYPES <  6
// DEQ
// DQT
// DQX
// DTQ
// DXQ
// DTS
// DHH
func TestDlin(t *testing.T) {
	var c2 string
	var eps, threq float64
	var i, j, kdmax, la, lafac, lda, nb, nmats, versMajor, versMinor, versPatch int

	nmax := 132
	maxrhs := 16
	nout := 6
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
	golapack.Ilaver(&versMajor, &versMinor, &versPatch)
	fmt.Printf(" Tests of the DOUBLE PRECISION LAPACK routines \n LAPACK VERSION %1d.%1d.%1d\n\n The following parameter values will be used:\n", versMajor, versMinor, versPatch)

	mval := []int{0, 1, 2, 3, 5, 10, 50}
	nm := len(mval)
	fmt.Printf("    %4s:", "M   ")
	for i = 1; i <= nm; i++ {
		fmt.Printf("  %6d", mval[i-1])
	}
	fmt.Printf("\n")

	nval := []int{0, 1, 2, 3, 5, 10, 50}
	nn := len(nval)
	fmt.Printf("    %4s:", "N   ")
	for i = 1; i <= nn; i++ {
		fmt.Printf("  %6d", nval[i-1])
	}
	fmt.Printf("\n")

	nsval := []int{1, 2, 15}
	nns := len(nsval)
	fmt.Printf("    %4s:", "NRHS")
	for i = 1; i <= nns; i++ {
		fmt.Printf("  %6d", nsval[i-1])
	}
	fmt.Printf("\n")

	nbval := []int{1, 3, 3, 3, 20}
	nnb := len(nbval)
	fmt.Printf("    %4s:", "NB  ")
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
	fmt.Printf("    %4s:", "NX  ")
	for i = 1; i <= nnb; i++ {
		fmt.Printf("  %6d", nxval[i-1])
	}
	fmt.Printf("\n")

	rankval := []int{30, 50, 90}
	nrank := 3
	fmt.Print("RANK % OF N:")
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

	for _, c2 = range []string{"DGE", "DGB", "DGT", "DPO", "DPS", "DPP", "DPB", "DPT", "DSY", "DSR", "DSK", "DSA", "DS2", "DSP", "DTR", "DTP", "DTB", "DQR", "DRQ", "DLQ", "DQL", "DQP", "DTZ", "DLS", "DEQ", "DQT", "DQX", "DXQ", "DTQ", "DTS", "DHH"} {
		switch c2 {
		case "DGE":
			//        GE:  general matrices
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkge(dotype, &nm, &mval, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvge(dotype, &nn, &nval, &nrhs, &thresh, tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DGB":
			//        GB:  general banded matrices
			la = (2*kdmax + 1) * nmax
			lafac = (3*kdmax + 1) * nmax
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkgb(&dotype, &nm, &mval, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, a[0].VectorIdx(0), &la, a[2].VectorIdx(0), &lafac, b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvgb(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, a[0].VectorIdx(0), &la, a[2].VectorIdx(0), &lafac, a[5].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DGT":
			//        GT:  general tridiagonal matrices
			nmats = 12
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkgt(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvgt(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DPO":
			//        PO:  positive definite matrices
			nmats = 9
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkpo(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvpo(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DPS":
			//        PS:  positive semi-definite matrices
			nmats = 9

			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkps(&dotype, &nn, &nval, &nnb2, &nbval2, &nrank, &rankval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), &piv, work.VectorIdx(0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DPP":
			//        PP:  positive definite packed matrices
			nmats = 9
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkpp(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvpp(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DPB":
			//        PB:  positive definite banded matrices
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkpb(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvpb(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), s, work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DPT":
			//        PT:  positive definite tridiagonal matrices
			nmats = 12
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkpt(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvpt(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DSY":
			//        SY:  symmetric indefinite matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchksy(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvsy(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DSR":
			//        SR:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				DchksyRook(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				DdrvsyRook(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DSK":
			//        SK:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			//             differnet matrix storage format than SR path version.
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				DchksyRk(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), e, a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				DdrvsyRk(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), e, a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DSA":
			//        SA:  symmetric indefinite matrices,
			//             with partial (Aasen's) pivoting algorithm
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				DchksyAa(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				DdrvsyAa(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DS2":
			//        SA:  symmetric indefinite matrices,
			//             with partial (Aasen's) pivoting algorithm
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				DchksyAa2stage(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				DdrvsyAa2stage(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DSP":
			//        SP:  symmetric indefinite packed matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchksp(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Ddrvsp(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DTR":
			//        TR:  triangular matrices
			nmats = 18
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchktr(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DTP":
			//        TP:  triangular packed matrices
			nmats = 18
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchktp(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DTB":
			//        TB:  triangular banded matrices
			nmats = 17
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchktb(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DQR":
			//        QR:  QR factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkqr(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DLQ":
			//        LQ:  LQ factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchklq(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DQL":
			//        QL:  QL factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkql(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DRQ":
			//        RQ:  RQ factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkrq(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), b[3].VectorIdx(0), work.VectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DQP":
			//        QP:  QR factorization with pivoting
			nmats = 6
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchkq3(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &thresh, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DTZ":
			//        TZ:  Trapezoidal matrix
			nmats = 3
			Alareq(&nmats, &dotype)

			if tstchk {
				Dchktz(&dotype, &nm, &mval, &nn, &nval, &thresh, &tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[2].VectorIdx(0), work.VectorIdx(0), &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DLS":
			//        LS:  Least squares drivers
			nmats = 6
			Alareq(&nmats, &dotype)

			if tstdrv {
				Ddrvls(&dotype, &nm, &mval, &nn, &nval, &nns, &nsval, &nnb, &nbval, &nxval, &thresh, &tsterr, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), b[2].VectorIdx(0), rwork, rwork.Off(nmax+1-1), &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "DEQ":
			//        EQ:  Equilibration routines for general and positive definite
			//             matrices (THREQ should be between 2 and 10)
			if tstchk {
				Dchkeq(&threq, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DQT":
			//        QT:  QRT routines for general matrices
			if tstchk {
				Dchkqrt(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DQX":
			//        QX:  QRT routines for triangular-pentagonal matrices
			if tstchk {
				Dchkqrtp(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DTQ":
			//        TQ:  LQT routines for general matrices
			if tstchk {
				Dchklqt(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DXQ":
			//        XQ:  LQT routines for triangular-pentagonal matrices
			if tstchk {
				Dchklqtp(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DTS":
			//        TS:  QR routines for tall-skinny matrices
			if tstchk {
				Dchktsqr(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "DHH":
			//        HH:  Householder reconstruction for tall-skinny matrices
			if tstchk {
				DchkorhrCol(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
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
// ZGE   11               List types on next line if 0 < NTYPES < 11
// ZGB    8               List types on next line if 0 < NTYPES <  8
// ZGT   12               List types on next line if 0 < NTYPES < 12
// ZPO    9               List types on next line if 0 < NTYPES <  9
// ZPS    9               List types on next line if 0 < NTYPES <  9
// ZPP    9               List types on next line if 0 < NTYPES <  9
// ZPB    8               List types on next line if 0 < NTYPES <  8
// ZPT   12               List types on next line if 0 < NTYPES < 12
// ZHE   10               List types on next line if 0 < NTYPES < 10
// ZHR   10               List types on next line if 0 < NTYPES < 10
// ZHK   10               List types on next line if 0 < NTYPES < 10
// ZHA   10               List types on next line if 0 < NTYPES < 10
// ZH2   10               List types on next line if 0 < NTYPES < 10
// ZSA   11               List types on next line if 0 < NTYPES < 10
// ZS2   11               List types on next line if 0 < NTYPES < 10
// ZHP   10               List types on next line if 0 < NTYPES < 10
// ZSY   11               List types on next line if 0 < NTYPES < 11
// ZSR   11               List types on next line if 0 < NTYPES < 11
// ZSK   11               List types on next line if 0 < NTYPES < 11
// ZSP   11               List types on next line if 0 < NTYPES < 11
// ZTR   18               List types on next line if 0 < NTYPES < 18
// ZTP   18               List types on next line if 0 < NTYPES < 18
// ZTB   17               List types on next line if 0 < NTYPES < 17
// ZQR    8               List types on next line if 0 < NTYPES <  8
// ZRQ    8               List types on next line if 0 < NTYPES <  8
// ZLQ    8               List types on next line if 0 < NTYPES <  8
// ZQL    8               List types on next line if 0 < NTYPES <  8
// ZQP    6               List types on next line if 0 < NTYPES <  6
// ZTZ    3               List types on next line if 0 < NTYPES <  3
// ZLS    6               List types on next line if 0 < NTYPES <  6
// ZEQ
// ZQT
// ZQX
// ZTS
// ZHH
func TestZlin(t *testing.T) {
	var c2 string
	var tstchk, tstdrv, tsterr bool
	var eps, threq, thresh float64
	var i, j, kdmax, la, lafac, lda, maxrhs, nb, nm, nmats, nmax, nn, nnb, nnb2, nns, nout, nrank, nrhs, versMajor, versMinor, versPatch int

	nmax = 132
	maxrhs = 16
	nout = 6
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
	golapack.Ilaver(&versMajor, &versMinor, &versPatch)
	fmt.Printf(" Tests of the COMPLEX*16 LAPACK routines \n LAPACK VERSION %1d.%1d.%1d\n\n The following parameter values will be used:\n", versMajor, versMinor, versPatch)

	mval = []int{0, 1, 2, 3, 5, 10, 50}
	nm = len(mval)
	fmt.Printf("    %4s:", "M   ")
	for i = 1; i <= nm; i++ {
		fmt.Printf("  %6d", mval[i-1])
	}
	fmt.Printf("\n")

	nval = []int{0, 1, 2, 3, 5, 10, 50}
	nn = len(nval)
	fmt.Printf("    %4s:", "N   ")
	for i = 1; i <= nn; i++ {
		fmt.Printf("  %6d", nval[i-1])
	}
	fmt.Printf("\n")

	nsval = []int{1, 2, 15}
	nns = len(nsval)
	fmt.Printf("    %4s:", "NRHS")
	for i = 1; i <= nns; i++ {
		fmt.Printf("  %6d", nsval[i-1])
	}
	fmt.Printf("\n")

	nbval = []int{1, 3, 3, 3, 20}
	nnb = len(nbval)
	fmt.Printf("    %4s:", "NB  ")
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
	fmt.Printf("    %4s:", "NX  ")
	for i = 1; i <= nnb; i++ {
		fmt.Printf("  %6d", nxval[i-1])
	}
	fmt.Printf("\n")

	rankval = []int{30, 50, 90}
	nrank = 3
	fmt.Print("RANK % OF N:")
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
	// []string{"ZGE", "ZGB", "ZGT", "ZPO", "ZPS", "ZPP", "ZPB", "ZPT", "ZHE", "ZHR", "ZHK", "ZHA", "ZH2", "ZSA", "ZS2", "ZHP", "ZSY", "ZSR", "ZSK", "ZSP", "ZTR", "ZTP", "ZTB", "ZQR", "ZRQ", "ZLQ", "ZQL", "ZQP", "ZTZ", "ZLS", "ZEQ", "ZQT", "ZQX", "ZXQ", "ZTQ", "ZTS", "ZHH"}
	for _, c2 = range []string{"ZGE", "ZGB", "ZGT", "ZPO", "ZPS", "ZPP", "ZPB", "ZPT", "ZHE", "ZHR", "ZHK", "ZHA", "ZH2", "ZSA", "ZS2", "ZHP", "ZSY", "ZSR", "ZSK", "ZSP", "ZTR", "ZTP", "ZTB", "ZQR", "ZRQ", "ZLQ", "ZQL", "ZQP", "ZTZ", "ZLS", "ZEQ", "ZQT", "ZQX", "ZXQ", "ZTQ", "ZTS", "ZHH"} {
		switch c2 {
		case "ZGE":
			//        GE:  general matrices
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkge(&dotype, &nm, &mval, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvge(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVectorIdx(0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZGB":
			//        GB:  general banded matrices
			la = (2*kdmax + 1) * nmax
			lafac = (3*kdmax + 1) * nmax
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkgb(&dotype, &nm, &mval, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, a[0].CVector(0, 0), &la, a[2].CVector(0, 0), &lafac, b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvgb(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, a[0].CVector(0, 0), &la, a[2].CVector(0, 0), &lafac, a[5].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZGT":
			//        GT:  general tridiagonal matrices
			nmats = 12
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkgt(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvgt(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZPO":
			//        PO:  positive definite matrices
			nmats = 9
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkpo(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvpo(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZPS":
			//        PS:  positive semi-definite matrices
			nmats = 9

			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkps(&dotype, &nn, &nval, &nnb2, &nbval2, &nrank, &rankval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), &piv, work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZPP":
			//        PP:  positive definite packed matrices
			nmats = 9
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkpp(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvpp(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZPB":
			//        PB:  positive definite banded matrices
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkpb(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvpb(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), s, work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZPT":
			//        PT:  positive definite tridiagonal matrices
			nmats = 12
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkpt(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, a[0].CVector(0, 0), s, a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvpt(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, a[0].CVector(0, 0), s, a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZHE":
			//        HE:  Hermitian indefinite matrices
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkhe(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvhe(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZHR":
			//        HR:  Hermitian indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkherook(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvherook(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZHK":
			//        HK:  Hermitian indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			//             different matrix storage format than HR path version.
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkherk(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvherk(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZHA":
			//        HA:  Hermitian matrices,
			//             Aasen Algorithm
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkheaa(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvheaa(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZH2":
			//        H2:  Hermitian matrices,
			//             with partial (Aasen's) pivoting algorithm
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkheaa2stage(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvheaa2stage(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZHP":
			//        HP:  Hermitian indefinite packed matrices
			nmats = 10
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkhp(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvhp(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZSY":
			//        SY:  symmetric indefinite matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchksy(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvsy(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZSR":
			//        SR:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchksyrook(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvsyrook(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZSK":
			//        SK:  symmetric indefinite matrices,
			//             with bounded Bunch-Kaufman (rook) pivoting algorithm,
			//             different matrix storage format than SR path version.
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchksyrk(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvsyrk(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), e, a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZSA":
			//        SA:  symmetric indefinite matrices with Aasen's algorithm,
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchksyaa(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvsyaa(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZS2":
			//        S2:  symmetric indefinite matrices with Aasen's algorithm
			//             2 stage
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchksyaa2stage(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvsyaa2stage(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZSP":
			//        SP:  symmetric indefinite packed matrices,
			//             with partial (Bunch-Kaufman) pivoting algorithm
			nmats = 11
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchksp(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

			if tstdrv {
				Zdrvsp(&dotype, &nn, &nval, &nrhs, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s driver routines were not tested\n", path)
			}

		case "ZTR":
			//        TR:  triangular matrices
			nmats = 18
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchktr(&dotype, &nn, &nval, &nnb2, &nbval2, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZTP":
			//        TP:  triangular packed matrices
			nmats = 18
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchktp(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZTB":
			//        TB:  triangular banded matrices
			nmats = 17
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchktb(&dotype, &nn, &nval, &nns, &nsval, &thresh, &tsterr, &lda, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZQR":
			//        QR:  QR factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkqr(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZLQ":
			//        LQ:  LQ factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchklq(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZQL":
			//        QL:  QL factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkql(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZRQ":
			//        RQ:  RQ factorization
			nmats = 8
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkrq(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &nrhs, &thresh, &tsterr, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZEQ":
			//        EQ:  Equilibration routines for general and positive definite
			//             matrices (THREQ should be between 2 and 10)
			if tstchk {
				Zchkeq(&threq, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZTZ":
			//        TZ:  Trapezoidal matrix
			nmats = 3
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchktz(&dotype, &nm, &mval, &nn, &nval, &thresh, &tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), s, b[0].CVector(0, 0), work.CVector(0, 0), rwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZQP":
			//        QP:  QR factorization with pivoting
			nmats = 6
			Alareq(&nmats, &dotype)

			if tstchk {
				Zchkq3(&dotype, &nm, &mval, &nn, &nval, &nnb, &nbval, &nxval, &thresh, a[0].CVector(0, 0), a[1].CVector(0, 0), s, b[0].CVector(0, 0), work.CVector(0, 0), rwork, &iwork, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZLS":
			//        LS:  Least squares drivers
			nmats = 6
			Alareq(&nmats, &dotype)

			if tstdrv {
				Zdrvls(&dotype, &nm, &mval, &nn, &nval, &nns, &nsval, &nnb, &nbval, &nxval, &thresh, &tsterr, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), s, s.Off(nmax+1-1), &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZQT":
			//        QT:  QRT routines for general matrices
			if tstchk {
				Zchkqrt(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZQX":
			//        QX:  QRT routines for triangular-pentagonal matrices
			if tstchk {
				Zchkqrtp(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZTQ":
			//        TQ:  LQT routines for general matrices
			if tstchk {
				Zchklqt(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZXQ":
			//        XQ:  LQT routines for triangular-pentagonal matrices
			if tstchk {
				Zchklqtp(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZTS":
			//        TS:  QR routines for tall-skinny matrices
			if tstchk {
				Zchktsqr(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		case "ZHH":
			//        HH:  Householder reconstruction for tall-skinny matrices
			if tstchk {
				Zchkunhrcol(&thresh, &tsterr, &nm, &mval, &nn, &nval, &nnb, &nbval, &nout, t)
			} else {
				fmt.Printf("\n %3s routines were not tested\n", path)
			}

		default:

			fmt.Printf("\n %3s:  Unrecognized path name\n", path)
		}
	}

	//     Branch to this line when the last record is read.

	fmt.Printf("\n End of tests\n")
}
