package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"golinalg/util"
	"testing"
)

// Dchkee tests the DOUBLE PRECISION LAPACK subroutines for the matrix
// eigenvalue problem.  The test paths in this version are
//
// NEP (Nonsymmetric Eigenvalue Problem):
//     Test DGEHRD, DORGHR, DHSEQR, DTREVC, DHSEIN, and DORMHR
//
// SEP (Symmetric Eigenvalue Problem):
//     Test DSYTRD, DORGTR, DSTEQR, DSTERF, DSTEIN, DSTEDC,
//     and drivers DSYEV(X), DSBEV(X), DSPEV(X), DSTEV(X),
//                 DSYEVD,   DSBEVD,   DSPEVD,   DSTEVD
//
// SVD (Singular Value Decomposition):
//     Test DGEBRD, DORGBR, DBDSQR, DBDSDC
//     and the drivers DGESVD, DGESDD
//
// DEV (Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test DGEEV
//
// DES (Nonsymmetric Schur form Driver):
//     Test DGEES
//
// DVX (Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test DGEEVX
//
// DSX (Nonsymmetric Schur form Expert Driver):
//     Test DGEESX
//
// DGG (Generalized Nonsymmetric Eigenvalue Problem):
//     Test DGGHD3, DGGBAL, DGGBAK, DHGEQZ, and DTGEVC
//
// DGS (Generalized Nonsymmetric Schur form Driver):
//     Test DGGES
//
// DGV (Generalized Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test DGGEV
//
// DGX (Generalized Nonsymmetric Schur form Expert Driver):
//     Test DGGESX
//
// DXV (Generalized Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test DGGEVX
//
// DSG (Symmetric Generalized Eigenvalue Problem):
//     Test DSYGST, DSYGV, DSYGVD, DSYGVX, DSPGST, DSPGV, DSPGVD,
//     DSPGVX, DSBGST, DSBGV, DSBGVD, and DSBGVX
//
// DSB (Symmetric Band Eigenvalue Problem):
//     Test DSBTRD
//
// DBB (Band Singular Value Decomposition):
//     Test DGBBRD
//
// DEC (Eigencondition estimation):
//     Test DLALN2, DLASY2, DLAEQU, DLAEXC, DTRSYL, DTREXC, DTRSNA,
//     DTRSEN, and DLAQTR
//
// DBL (Balancing a general matrix)
//     Test DGEBAL
//
// DBK (Back transformation on a balanced matrix)
//     Test DGEBAK
//
// DGL (Balancing a matrix pair)
//     Test DGGBAL
//
// DGK (Back transformation on a matrix pair)
//     Test DGGBAK
//
// GLM (Generalized Linear Regression Model):
//     Tests DGGGLM
//
// GQR (Generalized QR and RQ factorizations):
//     Tests DGGQRF and DGGRQF
//
// GSV (Generalized Singular Value Decomposition):
//     Tests DGGSVD, DGGSVP, DTGSJA, DLAGS2, DLAPLL, and DLAPMT
//
// CSD (CS decomposition):
//     Tests DORCSD
//
// LSE (Constrained Linear Least Squares):
//     Tests DGGLSE
//
// Each test path has a different set of inputs, but the data sets for
// the driver routines xEV, xES, xVX, and xSX can be concatenated in a
// single input file.  The first line of input should contain one of the
// 3-character path names in columns 1-3.  The number of remaining lines
// depends on what is found on the first line.
//
// The number of matrix types used in testing is often controllable from
// the input file.  The number of matrix types for each path, and the
// test routine that describes them, is as follows:
func TestDeig(t *testing.T) {
	var tstchk, tstdif, tstdrv, tsterr bool
	var eps, thresh, thrshn float64
	var i, info, k, liwork, lwork, maxtyp, ncmax, newsd, nk, nmax, nn, nout, nparms, nrhs, ntypes, versMajor, versMinor, versPatch int
	var nin *util.Reader
	dotype := make([]bool, 30)
	logwrk := make([]bool, 132)
	intstr := make([]byte, 10)
	iacc22 := make([]int, 20)
	inibl := make([]int, 20)
	inmin := make([]int, 20)
	inwin := make([]int, 20)
	ioldsd := []int{0, 0, 0, 1}
	iseed := make([]int, 4)
	ishfts := make([]int, 20)
	kval := make([]int, 20)
	mval := make([]int, 20)
	mxbval := make([]int, 20)
	nbcol := make([]int, 20)
	nbmin := make([]int, 20)
	nbval := make([]int, 20)
	nsval := make([]int, 20)
	nval := make([]int, 20)
	nxval := make([]int, 20)
	pval := make([]int, 20)

	iparms := &gltest.Common.Claenv.Iparms
	selval := &gltest.Common.Sslct.Selval
	selwi := &gltest.Common.Sslct.Selwi
	selwr := &gltest.Common.Sslct.Selwr
	*iparms = make([]int, 100)
	*selval = make([]bool, 20)
	*selwi = vf(20)
	*selwr = vf(20)

	maxtyp = 21
	nout = 6
	nmax = 132
	ncmax = 20
	lwork = nmax*(5*nmax+5) + 1
	liwork = nmax * (5*nmax + 20)
	iwork := make([]int, liwork)
	a := func() []*mat.Matrix {
		arr := make([]*mat.Matrix, 14)
		for u := range arr {
			arr[u] = mf(nmax, nmax, opts)
		}
		return arr
	}()
	b := func() []*mat.Matrix {
		arr := make([]*mat.Matrix, 5)
		for u := range arr {
			arr[u] = mf(nmax, nmax, opts)
		}
		return arr
	}()
	c := mf(ncmax*ncmax, ncmax*ncmax, opts)
	d := func() []*mat.Vector {
		arr := make([]*mat.Vector, 12)
		for u := range arr {
			arr[u] = vf(nmax)
		}
		return arr
	}()
	taua := vf(132)
	taub := vf(132)
	work := vf(lwork)
	x := vf(5 * nmax)
	result := vf(500)

	intstr[0], intstr[1], intstr[2], intstr[3], intstr[4], intstr[5], intstr[6], intstr[7], intstr[8], intstr[9] = '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
	ioldsd[0], ioldsd[1], ioldsd[2], ioldsd[3] = 0, 0, 0, 1

	for _, path := range []string{"NEP", "SEP", "SE2", "SVD", "DEC", "DEV", "DES", "DVX", "DSX", "DGG", "DGS", "DGV", "DGX", "DXV", "DSB", "DSG", "DBL", "DBK", "DGL", "DGK", "DBB", "GLM", "GQR", "GSV", "CSD", "LSE"} {
		c3 := []byte(path)
		tstchk = false
		tstdrv = false
		tsterr = false
		dgx := string(path) == "DGX"
		dxv := string(path) == "DXV"

		golapack.Ilaver(&versMajor, &versMinor, &versPatch)
		fmt.Printf("\n LAPACK VERSION %1d.%1d.%1d\n", versMajor, versMinor, versPatch)
		fmt.Printf("\n The following parameter values will be used:\n")

		//     Calculate and print the machine dependent constants.
		fmt.Printf("\n")
		eps = golapack.Dlamch(Underflow)
		fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "underflow", eps)
		eps = golapack.Dlamch(Overflow)
		fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "overflow ", eps)
		eps = golapack.Dlamch(Epsilon)
		fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "precision", eps)

		for i = 1; i <= 4; i++ {
			iseed[i-1] = ioldsd[i-1]
		}

		if string(c3) == "DHS" || string(c3) == "NEP" {
			//        -------------------------------------
			//        NEP:  Nonsymmetric Eigenvalue Problem
			//        -------------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			//           NS    = number of shifts
			//           MAXB  = minimum submatrix size

			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem routines\n")
			nval = []int{0, 1, 2, 3, 5, 10, 16}
			nbval = []int{1, 3, 3, 3, 20}
			nbmin = []int{2, 2, 2, 2, 2}
			nxval = []int{1, 0, 5, 9, 1}
			inmin = []int{11, 12, 11, 15, 11}
			inwin = []int{2, 3, 5, 3, 2}
			inibl = []int{0, 5, 7, 3, 200}
			ishfts = []int{1, 2, 4, 2, 1}
			iacc22 = []int{0, 1, 2, 0, 1}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 21
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Derrhs([]byte("DHSEQR"), t)
			}
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])
				Xlaenv(12, maxint(11, inmin[i-1]))
				Xlaenv(13, inwin[i-1])
				Xlaenv(14, inibl[i-1])
				Xlaenv(15, ishfts[i-1])
				Xlaenv(16, iacc22[i-1])
				//
				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NX =%4d, INMIN=%4d, INWIN =%4d, INIBL =%4d, ISHFTS =%4d, IACC22 =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1], maxint(11, inmin[i-1]), inwin[i-1], inibl[i-1], ishfts[i-1], iacc22[i-1])
				Dchkhs(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[4], &nmax, a[5], a[6], d[0], d[1], d[2], d[3], d[4], d[5], a[7], a[8], a[9], a[10], a[11], d[6], work, &lwork, &iwork, &logwrk, result, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "DCHKHS", info)
				}
			}

		} else if string(c3) == "DST" || string(c3) == "SEP" || string(c3) == "SE2" {
			//        ----------------------------------
			//        SEP:  Symmetric Eigenvalue Problem
			//        ----------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			fmt.Printf(" Tests of the Symmetric Eigenvalue Problem routines\n")
			nval = []int{0, 1, 2, 3, 5, 20}
			nbval = []int{1, 3, 3, 3, 10}
			nbmin = []int{2, 2, 2, 2, 2}
			nxval = []int{1, 0, 5, 9, 1}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 50.0
			tstchk = true
			tstdrv = true
			tsterr = true
			newsd = 1
			ntypes = 21
			Alareq(&ntypes, &dotype)
			dotype[8] = false
			ntypes = minint(maxtyp, ntypes)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			Xlaenv(9, 25)
			if tsterr {
				Derrst([]byte("DST"), t)
			}
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NB =%4d, NBMIN =%4d, NX =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					if string(c3) == "SE2" {
						(*&dotype)[8] = false
						Dchkst2stg(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1].VectorIdx(0), d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], a[2], &nmax, a[3], a[4].VectorIdx(0), d[11], a[5], work, &lwork, &iwork, &liwork, result, &info)
					} else {
						Dchkst(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1].VectorIdx(0), d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], a[2], &nmax, a[3], a[4].VectorIdx(0), d[11], a[5], work, &lwork, &iwork, &liwork, result, &info, t)
					}
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "DCHKST", info)
					}
				}
				if tstdrv {
					if string(c3) == "SE2" {
						(*&dotype)[8] = false
						Ddrvst2stg(&nn, &nval, func() *int { y := 18; return &y }(), &dotype, &iseed, &thresh, &nout, a[0], &nmax, d[2], d[3], d[4], d[5], d[7], d[8], d[9], d[10], a[1], &nmax, a[2], d[11], a[3], work, &lwork, &iwork, &liwork, result, &info, t)
					} else {
						Ddrvst(&nn, &nval, func() *int { y := 18; return &y }(), &dotype, &iseed, &thresh, &nout, a[0], &nmax, d[2], d[3], d[4], d[5], d[7], d[8], d[9], d[10], a[1], &nmax, a[2], d[11], a[3], work, &lwork, &iwork, &liwork, result, &info, t)
					}
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "DDRVST", info)
					}
				}
			}

		} else if string(c3) == "DSG" {
			//        ----------------------------------------------
			//        DSG:  Symmetric Generalized Eigenvalue Problem
			//        ----------------------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			fmt.Printf(" Tests of the Symmetric Eigenvalue Problem routines\n")
			nval = []int{0, 1, 2, 3, 5, 10, 16}
			nbval = []int{1, 3, 20}
			nbmin = []int{2, 2, 2}
			nxval = []int{1, 1, 1}
			nsval = []int{0, 0, 0}
			mxbval = []int{0, 0, 0}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 20.0
			tstchk = true
			tstdrv = true
			tsterr = true
			newsd = 1
			ntypes = 21
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(9, 25)
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NB =%4d, NBMIN =%4d, NX =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					//               CALL DDRVSG( NN, NVAL, MAXTYP, DOTYPE, ISEED, THRESH,
					//     $                      NOUT, A( 1, 1 ), NMAX, A( 1, 2 ), NMAX,
					//     $                      D( 1, 3 ), A( 1, 3 ), NMAX, A( 1, 4 ),
					//     $                      A( 1, 5 ), A( 1, 6 ), A( 1, 7 ), WORK,
					//     $                      LWORK, IWORK, LIWORK, RESULT, INFO )
					Ddrvsg2stg(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], &nmax, d[2], d[2], a[2], &nmax, a[3], a[4], a[5].VectorIdx(0), a[6].VectorIdx(0), work, &lwork, &iwork, &liwork, result, &info, t)
					if info != 0 {
						fmt.Printf(" *** Error code from %s = %4d\n", "DDRVSG", info)
					}
				}
			}

		} else if string(c3) == "DBD" || string(c3) == "SVD" {
			//        ----------------------------------
			//        SVD:  Singular Value Decomposition
			//        ----------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			//           NRHS  = number of right hand sides
			fmt.Printf(" Tests of the Singular Value Decomposition routines\n")
			mval = []int{0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 10, 10, 16, 16, 30, 30, 40, 40}
			nval = []int{0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 3, 10, 16, 10, 16, 30, 40, 30, 40}
			nbval = []int{1, 3, 3, 3, 20}
			nbmin = []int{2, 2, 2, 2, 2}
			nxval = []int{1, 0, 5, 9, 1}
			nsval = []int{2, 0, 2, 2, 2}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 50.0
			tstchk = true
			tstdrv = true
			tsterr = true
			newsd = 1
			ntypes = 16
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			Xlaenv(9, 25)

			//        Test the error exits
			if tsterr && tstchk {
				Derrbd([]byte("DBD"), t)
			}
			if tsterr && tstdrv {
				Derred([]byte("DBD"), t)
			}

			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])
				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NB =%4d, NBMIN =%4d, NX =%4d, NRHS =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1], nrhs)
				if tstchk {
					Dchkbd(&nn, &mval, &nval, &maxtyp, &dotype, &nrhs, &iseed, &thresh, a[0], &nmax, d[0], d[1], d[2], d[3], a[1], &nmax, a[2], a[3], a[4], &nmax, a[5], &nmax, a[6], a[7], work, &lwork, &iwork, &nout, &info, t)
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "DCHKBD", info)
					}
				}
				if tstdrv {
					Ddrvbd(&nn, &mval, &nval, &maxtyp, &dotype, &iseed, &thresh, a[0], &nmax, a[1], &nmax, a[2], &nmax, a[3], a[4], a[5], d[0], d[1], d[2], work, &lwork, &iwork, &nout, &info, t)
				}
			}

		} else if string(c3) == "DEV" {
			//        --------------------------------------------
			//        DEV:  Nonsymmetric Eigenvalue Problem Driver
			//              DGEEV (eigenvalues and eigenvectors)
			//        --------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Driver\n    DGEEV (eigenvalues and eigevectors)\n")
			nval = []int{0, 1, 2, 3, 5, 10, 20}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 20.0
			tsterr = true
			newsd = 2
			ioldsd = []int{2518, 3899, 995, 397}
			ntypes = 21
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(12, maxint(11, inmin[0]))
			Xlaenv(13, inwin[0])
			Xlaenv(14, inibl[0])
			Xlaenv(15, ishfts[0])
			Xlaenv(16, iacc22[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derred(c3, t)
				}
				Ddrvev(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], d[0], d[1], d[2], d[3], a[2], &nmax, a[3], &nmax, a[4], &nmax, result, work, &lwork, &iwork, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "DGEEV", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if string(c3) == "DES" {
			//        --------------------------------------------
			//        DES:  Nonsymmetric Eigenvalue Problem Driver
			//              DGEES (Schur form)
			//        --------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Driver\n    DGEES (Schur form)\n")
			nval = []int{0, 1, 2, 3, 5, 10, 20}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 20.0
			tsterr = true
			newsd = 2
			ioldsd = []int{2518, 3899, 995, 397}
			ntypes = 21
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(12, maxint(11, inmin[0]))
			Xlaenv(13, inwin[0])
			Xlaenv(14, inibl[0])
			Xlaenv(15, ishfts[0])
			Xlaenv(16, iacc22[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derred(c3, t)
				}
				Ddrves(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], d[0], d[1], d[2], d[3], a[3], &nmax, result, work, &lwork, &iwork, &logwrk, &info, t)
				if info != 0 {
					fmt.Printf(" *** Error code from %s = %4d\n", "DGEES", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if string(c3) == "DVX" {
			//        --------------------------------------------------------------
			//        DVX:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              DGEEVX (eigenvalues, eigenvectors and condition numbers)
			//        --------------------------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    DGEEVX (eigenvalues, eigenvectors and condition numbers)\n")
			nval = []int{0, 1, 2, 3, 5, 10, 20}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 20.0
			tsterr = true
			newsd = 2
			ioldsd = []int{2518, 3899, 995, 397}
			ntypes = 21
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(12, maxint(11, inmin[0]))
			Xlaenv(13, inwin[0])
			Xlaenv(14, inibl[0])
			Xlaenv(15, ishfts[0])
			Xlaenv(16, iacc22[0])
			if ntypes < 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derred(c3, t)
				}
				Ddrvvx(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], d[0], d[1], d[2], d[3], a[2], &nmax, a[3], &nmax, a[4], &nmax, d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], result, work, &lwork, &iwork, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "DGEEVX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if string(c3) == "DSX" {
			//        ---------------------------------------------------
			//        DSX:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              DGEESX (Schur form and condition numbers)
			//        ---------------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    DGEESX (Schur form and condition numbers)\n")
			nval = []int{0, 1, 2, 3, 5, 10}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 20.0
			tsterr = true
			newsd = 2
			ioldsd = []int{2518, 3899, 995, 397}
			ntypes = 21
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(12, maxint(11, inmin[0]))
			Xlaenv(13, inwin[0])
			Xlaenv(14, inibl[0])
			Xlaenv(15, ishfts[0])
			Xlaenv(16, iacc22[0])
			if ntypes < 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derred(c3, t)
				}
				Alareq(&ntypes, &dotype)
				Ddrvsx(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], d[0], d[1], d[2], d[3], d[4], d[5], a[3], &nmax, a[4], result, work, &lwork, &iwork, &logwrk, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "DGEESX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if string(c3) == "DGG" {
			//
			//        -------------------------------------------------
			//        DGG:  Generalized Nonsymmetric Eigenvalue Problem
			//        -------------------------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NS    = number of shifts
			//           MAXB  = minimum submatrix size
			//           IACC22: structured matrix multiply
			//           NBCOL = minimum column dimension for blocks
			fmt.Printf("\n Tests of the Generalized Nonsymmetric Eigenvalue Problem routines\n")
			nval = []int{0, 1, 2, 3, 5, 10, 16}
			nbval = []int{1, 1, 2, 2}
			nbmin = []int{40, 40, 2, 2}
			nsval = []int{2, 4, 2, 4}
			mxbval = []int{40, 40, 2, 2}
			iacc22 = []int{1, 2, 1, 2}
			nbcol = []int{40, 40, 2, 2}
			nxval = []int{0, 0, 0, 0}
			inmin = []int{0, 0, 0, 0}
			inwin = []int{0, 0, 0, 0}
			inibl = []int{0, 0, 0, 0}
			ishfts = []int{0, 0, 0, 0}
			iacc22 = []int{0, 0, 0, 0}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 20.0
			tstchk = true
			tstdrv = false
			tsterr = true
			newsd = 1
			ntypes = 26
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tstchk && tsterr {
				Derrgg(c3, t)
			}
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(4, nsval[i-1])
				Xlaenv(8, mxbval[i-1])
				Xlaenv(16, iacc22[i-1])
				Xlaenv(5, nbcol[i-1])
				//
				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NB =%4d, NBMIN =%4d, NS =%4d, MAXB =%4d, IACC22 =%4d, NBCOL =%4d\n", c3, nbval[i-1], nbmin[i-1], nsval[i-1], mxbval[i-1], iacc22[i-1], nbcol[i-1])
				tstdif = false
				thrshn = 10.
				if tstchk {
					Dchkgg(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &tstdif, &thrshn, &nout, a[0], &nmax, a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], &nmax, a[9], a[10], a[11], d[0], d[1], d[2], d[3], d[4], d[5], a[12], a[13], work, &lwork, &logwrk, result, &info, t)
					if info != 0 {
						fmt.Printf(" *** Error code from %s = %4d\n", "DCHKGG", info)
					}
				}
			}

		} else if string(c3) == "DGS" {
			//        -------------------------------------------------
			//        DGS:  Generalized Nonsymmetric Eigenvalue Problem
			//              DGGES (Schur form)
			//        -------------------------------------------------
			fmt.Printf("\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver DGGES\n")
			nval = []int{2, 6, 10, 12, 20, 30}
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			mxbval = []int{1}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 26
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(4, nsval[0])
			Xlaenv(8, mxbval[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derrgg(c3, t)
				}
				Ddrges(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], d[0], d[1], d[2], work, &lwork, result, &logwrk, &info, t)
				if info != 0 {
					fmt.Printf(" *** Error code from %s = %4d\n", "DDRGES", info)
				}

				//     Blocked version
				Xlaenv(16, 2)
				Ddrges3(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], d[0], d[1], d[2], work, &lwork, result, &logwrk, &info, t)
				if info != 0 {
					fmt.Printf(" *** Error code from %s = %4d\n", "DDRGES3", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if dgx {
			//        -------------------------------------------------
			//        DGX:  Generalized Nonsymmetric Eigenvalue Problem
			//              DGGESX (Schur form and condition numbers)
			//        -------------------------------------------------
			fmt.Printf("\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver DGGESX\n")
			nn = 2
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			mxbval = []int{1}
			nparms = len(nbval)
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 5
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(4, nsval[0])
			Xlaenv(8, mxbval[0])
			if nn < 0 {
				fmt.Printf(" %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derrgg(c3, t)
				}

				Xlaenv(5, 2)
				Ddrgsx(&nn, &ncmax, &thresh, nin, &nout, a[0], &nmax, a[1], a[2], a[3], a[4], a[5], d[0], d[1], d[2], c, toPtr(ncmax*ncmax), a[11].VectorIdx(0), work, &lwork, &iwork, &liwork, &logwrk, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "DDRGSX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if string(c3) == "DGV" {
			//        -------------------------------------------------
			//        DGV:  Generalized Nonsymmetric Eigenvalue Problem
			//              DGGEV (Eigenvalue/vector form)
			//        -------------------------------------------------
			fmt.Printf("\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver DGGEV\n")
			nval = []int{2, 6, 8, 10, 15, 20}
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			mxbval = []int{1}
			nn = len(nval)
			nparms = len(nbval)
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 26
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(4, nsval[0])
			Xlaenv(8, mxbval[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derrgg(c3, t)
				}
				Ddrgev(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], a[8], &nmax, d[0], d[1], d[2], d[3], d[4], d[5], work, &lwork, result, &info, t)
				if info != 0 {
					fmt.Printf(" *** Error code from %s = %4d\n", "DDRGEV", info)
				}

				//     Blocked version
				Ddrgev3(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], a[8], &nmax, d[0], d[1], d[2], d[3], d[4], d[5], work, &lwork, result, &info, t)
				if info != 0 {
					fmt.Printf(" *** Error code from %s = %4d\n", "DDRGEV3", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if dxv {
			//        -------------------------------------------------
			//        DXV:  Generalized Nonsymmetric Eigenvalue Problem
			//              DGGEVX (eigenvalue/vector with condition numbers)
			//        -------------------------------------------------
			fmt.Printf("\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver DGGEVX\n")
			nn = 5
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			mxbval = []int{1}
			nparms = len(nbval)
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 2
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, nbval[0])
			Xlaenv(2, nbmin[0])
			Xlaenv(3, nxval[0])
			Xlaenv(4, nsval[0])
			Xlaenv(8, mxbval[0])
			if nn < 0 {
				fmt.Printf(" %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					Derrgg(c3, t)
				}

				Ddrgvx(&nn, &thresh, nin, &nout, a[0], &nmax, a[1], a[2], a[3], d[0], d[1], d[2], a[4], a[5], &(iwork[0]), &(iwork[1]), d[3], d[4], d[5], d[6], d[7], d[8], work, &lwork, toSlice(&iwork, 2), toPtr(liwork-2), result, &logwrk, &info, t)

				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "DDRGVX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if string(c3) == "DSB" {
			//        ------------------------------
			//        DSB:  Symmetric Band Reduction
			//        ------------------------------
			fmt.Printf(" Tests of DSBTRD\n (reduction of a symmetric band matrix to tridiagonal form)\n")
			nval = []int{5, 20}
			kval = []int{0, 1, 2, 5, 16}
			nk = len(kval)
			nn = len(nval)
			nparms = 0
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 15
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if tsterr {
				Derrst([]byte("DSB"), t)
			}
			//         CALL DCHKSB( NN, NVAL, NK, KVAL, MAXTYP, DOTYPE, ISEED, THRESH,
			//     $                NOUT, A( 1, 1 ), NMAX, D( 1, 1 ), D( 1, 2 ),
			//     $                A( 1, 2 ), NMAX, WORK, LWORK, RESULT, INFO )
			Dchksb2stg(&nn, &nval, &nk, &kval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, d[0], d[1], d[2], d[3], d[4], a[1], &nmax, work, &lwork, result, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "DCHKSB", info)
			}

		} else if string(c3) == "DBB" {
			//        ------------------------------
			//        DBB:  General Band Reduction
			//        ------------------------------
			fmt.Printf(" Tests of DGBBRD\n (reduction of a general band matrix to real bidiagonal form)\n")
			mval = []int{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 16, 16}
			nval = []int{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 10, 16, 10, 16}
			nn = len(nval)
			kval = []int{0, 1, 2, 3, 16}
			nk = len(kval)
			nsval = []int{1, 2}
			nparms = len(nsval)
			thresh = 20.0
			tsterr = false
			newsd = 1
			ntypes = 15
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NRHS =%4d\n", c3, nrhs)
				Dchkbb(&nn, &mval, &nval, &nk, &kval, &maxtyp, &dotype, &nrhs, &iseed, &thresh, &nout, a[0], &nmax, a[1].Off(0, 0).UpdateRows(2*nmax), toPtr(2*nmax), d[0], d[1], a[3], &nmax, a[4], &nmax, a[5], &nmax, a[6], work, &lwork, result, &info, t)
				if info != 0 {
					fmt.Printf(" *** Error code from %s = %4d\n", "DCHKBB", info)
				}
			}

		} else if string(c3) == "GLM" {
			//        -----------------------------------------
			//        GLM:  Generalized Linear Regression Model
			//        -----------------------------------------
			fmt.Printf("\n Tests of the Generalized Linear Regression Model routines\n")
			mval = []int{0, 5, 8, 15, 20, 40}
			pval = []int{9, 0, 15, 12, 15, 30}
			nval = []int{5, 5, 10, 25, 30, 40}
			nn = len(nval)
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Derrgg([]byte("GLM"), t)
			}
			Dckglm(&nn, &mval, &pval, &nval, &ntypes, &iseed, &thresh, &nmax, a[0].Vector(0, 0), a[1].Vector(0, 0), b[0].Vector(0, 0), b[1].Vector(0, 0), x, work, d[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "DCKGLM", info)
			}

		} else if string(c3) == "GQR" {
			//        ------------------------------------------
			//        GQR:  Generalized QR and RQ factorizations
			//        ------------------------------------------
			fmt.Printf("\n Tests of the Generalized QR and RQ routines\n")
			mval = []int{0, 3, 10}
			pval = []int{0, 5, 20}
			nval = []int{0, 3, 30}
			nn = len(nval)
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Derrgg([]byte("GQR"), t)
			}
			Dckgqr(&nn, &mval, &nn, &pval, &nn, &nval, &ntypes, &iseed, &thresh, &nmax, a[0], a[1], a[2], a[3], taua, b[0], b[1], b[2], b[3], b[4], taub, work, d[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "DCKGQR", info)
			}

		} else if string(c3) == "GSV" {
			//        ----------------------------------------------
			//        GSV:  Generalized Singular Value Decomposition
			//        ----------------------------------------------
			fmt.Printf("\n Tests of the Generalized Singular Value Decomposition routines\n")
			mval = []int{0, 5, 9, 10, 20, 12, 12, 40}
			pval = []int{4, 0, 12, 14, 10, 10, 20, 15}
			nval = []int{3, 10, 15, 12, 8, 20, 8, 20}
			nn = len(nval)
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Derrgg([]byte("GSV"), t)
			}
			Dckgsv(&nn, &mval, &pval, &nval, &ntypes, &iseed, &thresh, &nmax, a[0], a[1], b[0], b[1], a[2], b[2], a[3], taua, taub, b[3], &iwork, work, d[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "DCKGSV", info)
			}

		} else if string(c3) == "CSD" {
			//        ----------------------------------------------
			//        CSD:  CS Decomposition
			//        ----------------------------------------------
			fmt.Printf("\n Tests of the CS Decomposition routines\n")
			mval = []int{0, 10, 10, 10, 10, 21, 24, 30, 22, 32, 55}
			pval = []int{0, 4, 4, 0, 10, 9, 10, 20, 12, 12, 40}
			nval = []int{0, 0, 10, 4, 4, 15, 12, 8, 20, 8, 20}
			nn = len(mval)
			thresh = 30.0
			tsterr = true
			newsd = 1
			ntypes = 4
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Derrgg([]byte("CSD"), t)
			}
			Dckcsd(&nn, &mval, &pval, &nval, &ntypes, &iseed, &thresh, &nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), a[5].VectorIdx(0), a[6].VectorIdx(0), &iwork, work, d[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "DCKCSD", info)
			}

		} else if string(c3) == "LSE" {
			//        --------------------------------------
			//        LSE:  Constrained Linear Least Squares
			//        --------------------------------------
			//
			fmt.Printf("\n Tests of the Linear Least Squares routines\n")
			mval = []int{6, 0, 5, 8, 10, 30}
			pval = []int{0, 5, 5, 5, 8, 20}
			nval = []int{5, 5, 6, 8, 12, 40}
			nn = len(mval)
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			Alareq(&ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Derrgg([]byte("LSE"), t)
			}
			Dcklse(&nn, &mval, &pval, &nval, &ntypes, &iseed, &thresh, &nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), x, work, d[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "DCKLSE", info)
			}

		} else if string(c3) == "DBL" {
			//        DGEBAL:  Balancing
			Dchkbl(t)

		} else if string(c3) == "DBK" {
			//        DGEBAK:  Back transformation
			Dchkbk(t)

		} else if string(c3) == "DGL" {
			//        DGGBAL:  Balancing
			Dchkgl(t)

		} else if string(c3) == "DGK" {
			//        DGGBAK:  Back transformation
			Dchkgk(t)

		} else if string(c3) == "DEC" {
			//        DEC:  Eigencondition estimation
			Xlaenv(1, 1)
			Xlaenv(12, 11)
			Xlaenv(13, 2)
			Xlaenv(14, 0)
			Xlaenv(15, 2)
			Xlaenv(16, 2)
			thresh = 50.0
			tsterr = true
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Dchkec(&thresh, &tsterr, t)

		} else {
			fmt.Printf("\n")
			fmt.Printf("\n")
			fmt.Printf(" %3s:  Unrecognized path name\n", c3)
		}

	}
	fmt.Printf("\n\n End of tests\n")
}

// Zchkee tests the COMPLEX*16 LAPACK subroutines for the matrix
// eigenvalue problem.  The test paths in this version are
//
// NEP (Nonsymmetric Eigenvalue Problem):
//     Test ZGEHRD, ZUNGHR, ZHSEQR, ZTREVC, ZHSEIN, and ZUNMHR
//
// SEP (Hermitian Eigenvalue Problem):
//     Test ZHETRD, ZUNGTR, ZSTEQR, ZSTERF, ZSTEIN, ZSTEDC,
//     and drivers ZHEEV(X), ZHBEV(X), ZHPEV(X),
//                 ZHEEVD,   ZHBEVD,   ZHPEVD
//
// SVD (Singular Value Decomposition):
//     Test ZGEBRD, ZUNGBR, and ZBDSQR
//     and the drivers ZGESVD, ZGESDD
//
// ZEV (Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test ZGEEV
//
// ZES (Nonsymmetric Schur form Driver):
//     Test ZGEES
//
// ZVX (Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test ZGEEVX
//
// ZSX (Nonsymmetric Schur form Expert Driver):
//     Test ZGEESX
//
// ZGG (Generalized Nonsymmetric Eigenvalue Problem):
//     Test ZGGHD3, ZGGBAL, ZGGBAK, ZHGEQZ, and ZTGEVC
//
// ZGS (Generalized Nonsymmetric Schur form Driver):
//     Test ZGGES
//
// ZGV (Generalized Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test ZGGEV
//
// ZGX (Generalized Nonsymmetric Schur form Expert Driver):
//     Test ZGGESX
//
// ZXV (Generalized Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test ZGGEVX
//
// ZSG (Hermitian Generalized Eigenvalue Problem):
//     Test ZHEGST, ZHEGV, ZHEGVD, ZHEGVX, ZHPGST, ZHPGV, ZHPGVD,
//     ZHPGVX, ZHBGST, ZHBGV, ZHBGVD, and ZHBGVX
//
// ZHB (Hermitian Band Eigenvalue Problem):
//     Test ZHBTRD
//
// ZBB (Band Singular Value Decomposition):
//     Test ZGBBRD
//
// ZEC (Eigencondition estimation):
//     Test ZTRSYL, ZTREXC, ZTRSNA, and ZTRSEN
//
// ZBL (Balancing a general matrix)
//     Test ZGEBAL
//
// ZBK (Back transformation on a balanced matrix)
//     Test ZGEBAK
//
// ZGL (Balancing a matrix pair)
//     Test ZGGBAL
//
// ZGK (Back transformation on a matrix pair)
//     Test ZGGBAK
//
// GLM (Generalized Linear Regression Model):
//     Tests ZGGGLM
//
// GQR (Generalized QR and RQ factorizations):
//     Tests ZGGQRF and ZGGRQF
//
// GSV (Generalized Singular Value Decomposition):
//     Tests ZGGSVD, ZGGSVP, ZTGSJA, ZLAGS2, ZLAPLL, and ZLAPMT
//
// CSD (CS decomposition):
//     Tests ZUNCSD
//
// LSE (Constrained Linear Least Squares):
//     Tests ZGGLSE
//
// Each test path has a different set of inputs, but the data sets for
// the driver routines xEV, xES, xVX, and xSX can be concatenated in a
// single input file.  The first line of input should contain one of the
// 3-character path names in columns 1-3.  The number of remaining lines
// depends on what is found on the first line.
//
// The number of matrix types used in testing is often controllable from
// the input file.  The number of matrix types for each path, and the
// test routine that describes them, is as follows:
//
// Path name(s)  Types    Test routine
//
// ZHS or NEP      21     ZCHKHS
// ZST or SEP      21     ZCHKST (routines)
//                 18     ZDRVST (drivers)
// ZBD or SVD      16     ZCHKBD (routines)
//                  5     ZDRVBD (drivers)
// ZEV             21     ZDRVEV
// ZES             21     ZDRVES
// ZVX             21     ZDRVVX
// ZSX             21     ZDRVSX
// ZGG             26     ZCHKGG (routines)
// ZGS             26     ZDRGES
// ZGX              5     ZDRGSX
// ZGV             26     ZDRGEV
// ZXV              2     ZDRGVX
// ZSG             21     ZDRVSG
// ZHB             15     ZCHKHB
// ZBB             15     ZCHKBB
// ZEC              -     ZCHKEC
// ZBL              -     ZCHKBL
// ZBK              -     ZCHKBK
// ZGL              -     ZCHKGL
// ZGK              -     ZCHKGK
// GLM              8     ZCKGLM
// GQR              8     ZCKGQR
// GSV              8     ZCKGSV
// CSD              3     ZCKCSD
// LSE              8     ZCKLSE
//
//-----------------------------------------------------------------------
//
// NEP input file:
//
// line 2:  NN, INTEGER
//          Number of values of N.
//
// line 3:  NVAL, INTEGER array, dimension (NN)
//          The values for the matrix dimension N.
//
// line 4:  NPARMS, INTEGER
//          Number of values of the parameters NB, NBMIN, NX, NS, and
//          MAXB.
//
// line 5:  NBVAL, INTEGER array, dimension (NPARMS)
//          The values for the blocksize NB.
//
// line 6:  NBMIN, INTEGER array, dimension (NPARMS)
//          The values for the minimum blocksize NBMIN.
//
// line 7:  NXVAL, INTEGER array, dimension (NPARMS)
//          The values for the crossover point NX.
//
// line 8:  INMIN, INTEGER array, dimension (NPARMS)
//          LAHQR vs TTQRE crossover point, >= 11
//
// line 9:  INWIN, INTEGER array, dimension (NPARMS)
//          recommended deflation window size
//
// line 10: INIBL, INTEGER array, dimension (NPARMS)
//          nibble crossover point
//
// line 11:  ISHFTS, INTEGER array, dimension (NPARMS)
//          number of simultaneous shifts)
//
// line 12:  IACC22, INTEGER array, dimension (NPARMS)
//          select structured matrix multiply: 0, 1 or 2)
//
// line 13: THRESH
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.  To have all of the test
//          ratios printed, use THRESH = 0.0 .
//
// line 14: NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 14 was 2:
//
// line 15: INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 15-EOF:  The remaining lines occur in sets of 1 or 2 and allow
//          the user to specify the matrix types.  Each line contains
//          a 3-character path name in columns 1-3, and the number
//          of matrix types must be the first nonblank item in columns
//          4-80.  If the number of matrix types is at least 1 but is
//          less than the maximum number of possible types, a second
//          line will be read to get the numbers of the matrix types to
//          be used.  For example,
// NEP 21
//          requests all of the matrix types for the nonsymmetric
//          eigenvalue problem, while
// NEP  4
// 9 10 11 12
//          requests only matrices of _type 9, 10, 11, and 12.
//
//          The valid 3-character path names are 'NEP' or 'ZHS' for the
//          nonsymmetric eigenvalue routines.
//
//-----------------------------------------------------------------------
//
// SEP or ZSG input file:
//
// line 2:  NN, INTEGER
//          Number of values of N.
//
// line 3:  NVAL, INTEGER array, dimension (NN)
//          The values for the matrix dimension N.
//
// line 4:  NPARMS, INTEGER
//          Number of values of the parameters NB, NBMIN, and NX.
//
// line 5:  NBVAL, INTEGER array, dimension (NPARMS)
//          The values for the blocksize NB.
//
// line 6:  NBMIN, INTEGER array, dimension (NPARMS)
//          The values for the minimum blocksize NBMIN.
//
// line 7:  NXVAL, INTEGER array, dimension (NPARMS)
//          The values for the crossover point NX.
//
// line 8:  THRESH
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 9:  TSTCHK, LOGICAL
//          Flag indicating whether or not to test the LAPACK routines.
//
// line 10: TSTDRV, LOGICAL
//          Flag indicating whether or not to test the driver routines.
//
// line 11: TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 12: NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 12 was 2:
//
// line 13: INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 13-EOF:  Lines specifying matrix types, as for NEP.
//          The valid 3-character path names are 'SEP' or 'ZST' for the
//          Hermitian eigenvalue routines and driver routines, and
//          'ZSG' for the routines for the Hermitian generalized
//          eigenvalue problem.
//
//-----------------------------------------------------------------------
//
// SVD input file:
//
// line 2:  NN, INTEGER
//          Number of values of M and N.
//
// line 3:  MVAL, INTEGER array, dimension (NN)
//          The values for the matrix row dimension M.
//
// line 4:  NVAL, INTEGER array, dimension (NN)
//          The values for the matrix column dimension N.
//
// line 5:  NPARMS, INTEGER
//          Number of values of the parameter NB, NBMIN, NX, and NRHS.
//
// line 6:  NBVAL, INTEGER array, dimension (NPARMS)
//          The values for the blocksize NB.
//
// line 7:  NBMIN, INTEGER array, dimension (NPARMS)
//          The values for the minimum blocksize NBMIN.
//
// line 8:  NXVAL, INTEGER array, dimension (NPARMS)
//          The values for the crossover point NX.
//
// line 9:  NSVAL, INTEGER array, dimension (NPARMS)
//          The values for the number of right hand sides NRHS.
//
// line 10: THRESH
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 11: TSTCHK, LOGICAL
//          Flag indicating whether or not to test the LAPACK routines.
//
// line 12: TSTDRV, LOGICAL
//          Flag indicating whether or not to test the driver routines.
//
// line 13: TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 14: NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 14 was 2:
//
// line 15: INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 15-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path names are 'SVD' or 'ZBD' for both the
//          SVD routines and the SVD driver routines.
//
//-----------------------------------------------------------------------
//
// ZEV and ZES data files:
//
// line 1:  'ZEV' or 'ZES' in columns 1 to 3.
//
// line 2:  NSIZES, INTEGER
//          Number of sizes of matrices to use. Should be at least 0
//          and at most 20. If NSIZES = 0, no testing is done
//          (although the remaining  3 lines are still read).
//
// line 3:  NN, INTEGER array, dimension(NSIZES)
//          Dimensions of matrices to be tested.
//
// line 4:  NB, NBMIN, NX, NS, NBCOL, INTEGERs
//          These integer parameters determine how blocking is done
//          (see ILAENV for details)
//          NB     : block size
//          NBMIN  : minimum block size
//          NX     : minimum dimension for blocking
//          NS     : number of shifts in xHSEQR
//          NBCOL  : minimum column dimension for blocking
//
// line 5:  THRESH, REAL
//          The test threshold against which computed residuals are
//          compared. Should generally be in the range from 10. to 20.
//          If it is 0., all test case data will be printed.
//
// line 6:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 6 was 2:
//
// line 7:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 8 and following:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'ZEV' to test CGEEV, or
//          'ZES' to test CGEES.
//
//-----------------------------------------------------------------------
//
// The ZVX data has two parts. The first part is identical to ZEV,
// and the second part consists of test matrices with precomputed
// solutions.
//
// line 1:  'ZVX' in columns 1-3.
//
// line 2:  NSIZES, INTEGER
//          If NSIZES = 0, no testing of randomly generated examples
//          is done, but any precomputed examples are tested.
//
// line 3:  NN, INTEGER array, dimension(NSIZES)
//
// line 4:  NB, NBMIN, NX, NS, NBCOL, INTEGERs
//
// line 5:  THRESH, REAL
//
// line 6:  NEWSD, INTEGER
//
// If line 6 was 2:
//
// line 7:  INTEGER array, dimension (4)
//
// lines 8 and following: The first line contains 'ZVX' in columns 1-3
//          followed by the number of matrix types, possibly with
//          a second line to specify certain matrix types.
//          If the number of matrix types = 0, no testing of randomly
//          generated examples is done, but any precomputed examples
//          are tested.
//
// remaining lines : Each matrix is stored on 1+N+N**2 lines, where N is
//          its dimension. The first line contains the dimension N and
//          ISRT (two integers). ISRT indicates whether the last N lines
//          are sorted by increasing real part of the eigenvalue
//          (ISRT=0) or by increasing imaginary part (ISRT=1). The next
//          N**2 lines contain the matrix rowwise, one entry per line.
//          The last N lines correspond to each eigenvalue. Each of
//          these last N lines contains 4 real values: the real part of
//          the eigenvalues, the imaginary part of the eigenvalue, the
//          reciprocal condition number of the eigenvalues, and the
//          reciprocal condition number of the vector eigenvector. The
//          end of data is indicated by dimension N=0. Even if no data
//          is to be tested, there must be at least one line containing
//          N=0.
//
//-----------------------------------------------------------------------
//
// The ZSX data is like ZVX. The first part is identical to ZEV, and the
// second part consists of test matrices with precomputed solutions.
//
// line 1:  'ZSX' in columns 1-3.
//
// line 2:  NSIZES, INTEGER
//          If NSIZES = 0, no testing of randomly generated examples
//          is done, but any precomputed examples are tested.
//
// line 3:  NN, INTEGER array, dimension(NSIZES)
//
// line 4:  NB, NBMIN, NX, NS, NBCOL, INTEGERs
//
// line 5:  THRESH, REAL
//
// line 6:  NEWSD, INTEGER
//
// If line 6 was 2:
//
// line 7:  INTEGER array, dimension (4)
//
// lines 8 and following: The first line contains 'ZSX' in columns 1-3
//          followed by the number of matrix types, possibly with
//          a second line to specify certain matrix types.
//          If the number of matrix types = 0, no testing of randomly
//          generated examples is done, but any precomputed examples
//          are tested.
//
// remaining lines : Each matrix is stored on 3+N**2 lines, where N is
//          its dimension. The first line contains the dimension N, the
//          dimension M of an invariant subspace, and ISRT. The second
//          line contains M integers, identifying the eigenvalues in the
//          invariant subspace (by their position in a list of
//          eigenvalues ordered by increasing real part (if ISRT=0) or
//          by increasing imaginary part (if ISRT=1)). The next N**2
//          lines contain the matrix rowwise. The last line contains the
//          reciprocal condition number for the average of the selected
//          eigenvalues, and the reciprocal condition number for the
//          corresponding right invariant subspace. The end of data in
//          indicated by a line containing N=0, M=0, and ISRT = 0.  Even
//          if no data is to be tested, there must be at least one line
//          containing N=0, M=0 and ISRT=0.
//
//-----------------------------------------------------------------------
//
// ZGG input file:
//
// line 2:  NN, INTEGER
//          Number of values of N.
//
// line 3:  NVAL, INTEGER array, dimension (NN)
//          The values for the matrix dimension N.
//
// line 4:  NPARMS, INTEGER
//          Number of values of the parameters NB, NBMIN, NBCOL, NS, and
//          MAXB.
//
// line 5:  NBVAL, INTEGER array, dimension (NPARMS)
//          The values for the blocksize NB.
//
// line 6:  NBMIN, INTEGER array, dimension (NPARMS)
//          The values for NBMIN, the minimum row dimension for blocks.
//
// line 7:  NSVAL, INTEGER array, dimension (NPARMS)
//          The values for the number of shifts.
//
// line 8:  MXBVAL, INTEGER array, dimension (NPARMS)
//          The values for MAXB, used in determining minimum blocksize.
//
// line 9:  IACC22, INTEGER array, dimension (NPARMS)
//          select structured matrix multiply: 1 or 2)
//
// line 10: NBCOL, INTEGER array, dimension (NPARMS)
//          The values for NBCOL, the minimum column dimension for
//          blocks.
//
// line 11: THRESH
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 12: TSTCHK, LOGICAL
//          Flag indicating whether or not to test the LAPACK routines.
//
// line 13: TSTDRV, LOGICAL
//          Flag indicating whether or not to test the driver routines.
//
// line 14: TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 15: NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 15 was 2:
//
// line 16: INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 17-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'ZGG' for the generalized
//          eigenvalue problem routines and driver routines.
//
//-----------------------------------------------------------------------
//
// ZGS and ZGV input files:
//
// line 1:  'ZGS' or 'ZGV' in columns 1 to 3.
//
// line 2:  NN, INTEGER
//          Number of values of N.
//
// line 3:  NVAL, INTEGER array, dimension(NN)
//          Dimensions of matrices to be tested.
//
// line 4:  NB, NBMIN, NX, NS, NBCOL, INTEGERs
//          These integer parameters determine how blocking is done
//          (see ILAENV for details)
//          NB     : block size
//          NBMIN  : minimum block size
//          NX     : minimum dimension for blocking
//          NS     : number of shifts in xHGEQR
//          NBCOL  : minimum column dimension for blocking
//
// line 5:  THRESH, REAL
//          The test threshold against which computed residuals are
//          compared. Should generally be in the range from 10. to 20.
//          If it is 0., all test case data will be printed.
//
// line 6:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits.
//
// line 7:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 17 was 2:
//
// line 7:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 7-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'ZGS' for the generalized
//          eigenvalue problem routines and driver routines.
//
//-----------------------------------------------------------------------
//
// ZGX input file:
// line 1:  'ZGX' in columns 1 to 3.
//
// line 2:  N, INTEGER
//          Value of N.
//
// line 3:  NB, NBMIN, NX, NS, NBCOL, INTEGERs
//          These integer parameters determine how blocking is done
//          (see ILAENV for details)
//          NB     : block size
//          NBMIN  : minimum block size
//          NX     : minimum dimension for blocking
//          NS     : number of shifts in xHGEQR
//          NBCOL  : minimum column dimension for blocking
//
// line 4:  THRESH, REAL
//          The test threshold against which computed residuals are
//          compared. Should generally be in the range from 10. to 20.
//          Information will be printed about each test for which the
//          test ratio is greater than or equal to the threshold.
//
// line 5:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 6:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 6 was 2:
//
// line 7: INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// If line 2 was 0:
//
// line 7-EOF: Precomputed examples are tested.
//
// remaining lines : Each example is stored on 3+2*N*N lines, where N is
//          its dimension. The first line contains the dimension (a
//          single integer).  The next line contains an integer k such
//          that only the last k eigenvalues will be selected and appear
//          in the leading diagonal blocks of $A$ and $B$. The next N*N
//          lines contain the matrix A, one element per line. The next N*N
//          lines contain the matrix B. The last line contains the
//          reciprocal of the eigenvalue cluster condition number and the
//          reciprocal of the deflating subspace (associated with the
//          selected eigencluster) condition number.  The end of data is
//          indicated by dimension N=0.  Even if no data is to be tested,
//          there must be at least one line containing N=0.
//
//-----------------------------------------------------------------------
//
// ZXV input files:
// line 1:  'ZXV' in columns 1 to 3.
//
// line 2:  N, INTEGER
//          Value of N.
//
// line 3:  NB, NBMIN, NX, NS, NBCOL, INTEGERs
//          These integer parameters determine how blocking is done
//          (see ILAENV for details)
//          NB     : block size
//          NBMIN  : minimum block size
//          NX     : minimum dimension for blocking
//          NS     : number of shifts in xHGEQR
//          NBCOL  : minimum column dimension for blocking
//
// line 4:  THRESH, REAL
//          The test threshold against which computed residuals are
//          compared. Should generally be in the range from 10. to 20.
//          Information will be printed about each test for which the
//          test ratio is greater than or equal to the threshold.
//
// line 5:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 6:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 6 was 2:
//
// line 7: INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// If line 2 was 0:
//
// line 7-EOF: Precomputed examples are tested.
//
// remaining lines : Each example is stored on 3+2*N*N lines, where N is
//          its dimension. The first line contains the dimension (a
//          single integer). The next N*N lines contain the matrix A, one
//          element per line. The next N*N lines contain the matrix B.
//          The next line contains the reciprocals of the eigenvalue
//          condition numbers.  The last line contains the reciprocals of
//          the eigenvector condition numbers.  The end of data is
//          indicated by dimension N=0.  Even if no data is to be tested,
//          there must be at least one line containing N=0.
//
//-----------------------------------------------------------------------
//
// ZHB input file:
//
// line 2:  NN, INTEGER
//          Number of values of N.
//
// line 3:  NVAL, INTEGER array, dimension (NN)
//          The values for the matrix dimension N.
//
// line 4:  NK, INTEGER
//          Number of values of K.
//
// line 5:  KVAL, INTEGER array, dimension (NK)
//          The values for the matrix dimension K.
//
// line 6:  THRESH
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 7:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 7 was 2:
//
// line 8:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 8-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'ZHB'.
//
//-----------------------------------------------------------------------
//
// ZBB input file:
//
// line 2:  NN, INTEGER
//          Number of values of M and N.
//
// line 3:  MVAL, INTEGER array, dimension (NN)
//          The values for the matrix row dimension M.
//
// line 4:  NVAL, INTEGER array, dimension (NN)
//          The values for the matrix column dimension N.
//
// line 4:  NK, INTEGER
//          Number of values of K.
//
// line 5:  KVAL, INTEGER array, dimension (NK)
//          The values for the matrix bandwidth K.
//
// line 6:  NPARMS, INTEGER
//          Number of values of the parameter NRHS
//
// line 7:  NSVAL, INTEGER array, dimension (NPARMS)
//          The values for the number of right hand sides NRHS.
//
// line 8:  THRESH
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 9:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 9 was 2:
//
// line 10: INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 10-EOF:  Lines specifying matrix types, as for SVD.
//          The 3-character path name is 'ZBB'.
//
//-----------------------------------------------------------------------
//
// ZEC input file:
//
// line  2: THRESH, REAL
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// lines  3-EOF:
//
// Input for testing the eigencondition routines consists of a set of
// specially constructed test cases and their solutions.  The data
// format is not intended to be modified by the user.
//
//-----------------------------------------------------------------------
//
// ZBL and ZBK input files:
//
// line 1:  'ZBL' in columns 1-3 to test CGEBAL, or 'ZBK' in
//          columns 1-3 to test CGEBAK.
//
// The remaining lines consist of specially constructed test cases.
//
//-----------------------------------------------------------------------
//
// ZGL and ZGK input files:
//
// line 1:  'ZGL' in columns 1-3 to test ZGGBAL, or 'ZGK' in
//          columns 1-3 to test ZGGBAK.
//
// The remaining lines consist of specially constructed test cases.
//
//-----------------------------------------------------------------------
//
// GLM data file:
//
// line 1:  'GLM' in columns 1 to 3.
//
// line 2:  NN, INTEGER
//          Number of values of M, P, and N.
//
// line 3:  MVAL, INTEGER array, dimension(NN)
//          Values of M (row dimension).
//
// line 4:  PVAL, INTEGER array, dimension(NN)
//          Values of P (row dimension).
//
// line 5:  NVAL, INTEGER array, dimension(NN)
//          Values of N (column dimension), note M <= N <= M+P.
//
// line 6:  THRESH, REAL
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 7:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 8:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 8 was 2:
//
// line 9:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 9-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'GLM' for the generalized
//          linear regression model routines.
//
//-----------------------------------------------------------------------
//
// GQR data file:
//
// line 1:  'GQR' in columns 1 to 3.
//
// line 2:  NN, INTEGER
//          Number of values of M, P, and N.
//
// line 3:  MVAL, INTEGER array, dimension(NN)
//          Values of M.
//
// line 4:  PVAL, INTEGER array, dimension(NN)
//          Values of P.
//
// line 5:  NVAL, INTEGER array, dimension(NN)
//          Values of N.
//
// line 6:  THRESH, REAL
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 7:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 8:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 8 was 2:
//
// line 9:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 9-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'GQR' for the generalized
//          QR and RQ routines.
//
//-----------------------------------------------------------------------
//
// GSV data file:
//
// line 1:  'GSV' in columns 1 to 3.
//
// line 2:  NN, INTEGER
//          Number of values of M, P, and N.
//
// line 3:  MVAL, INTEGER array, dimension(NN)
//          Values of M (row dimension).
//
// line 4:  PVAL, INTEGER array, dimension(NN)
//          Values of P (row dimension).
//
// line 5:  NVAL, INTEGER array, dimension(NN)
//          Values of N (column dimension).
//
// line 6:  THRESH, REAL
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 7:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 8:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 8 was 2:
//
// line 9:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 9-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'GSV' for the generalized
//          SVD routines.
//
//-----------------------------------------------------------------------
//
// CSD data file:
//
// line 1:  'CSD' in columns 1 to 3.
//
// line 2:  NM, INTEGER
//          Number of values of M, P, and N.
//
// line 3:  MVAL, INTEGER array, dimension(NM)
//          Values of M (row and column dimension of orthogonal matrix).
//
// line 4:  PVAL, INTEGER array, dimension(NM)
//          Values of P (row dimension of top-left block).
//
// line 5:  NVAL, INTEGER array, dimension(NM)
//          Values of N (column dimension of top-left block).
//
// line 6:  THRESH, REAL
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 7:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 8:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 8 was 2:
//
// line 9:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 9-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'CSD' for the CSD routine.
//
//-----------------------------------------------------------------------
//
// LSE data file:
//
// line 1:  'LSE' in columns 1 to 3.
//
// line 2:  NN, INTEGER
//          Number of values of M, P, and N.
//
// line 3:  MVAL, INTEGER array, dimension(NN)
//          Values of M.
//
// line 4:  PVAL, INTEGER array, dimension(NN)
//          Values of P.
//
// line 5:  NVAL, INTEGER array, dimension(NN)
//          Values of N, note P <= N <= P+M.
//
// line 6:  THRESH, REAL
//          Threshold value for the test ratios.  Information will be
//          printed about each test for which the test ratio is greater
//          than or equal to the threshold.
//
// line 7:  TSTERR, LOGICAL
//          Flag indicating whether or not to test the error exits for
//          the LAPACK routines and driver routines.
//
// line 8:  NEWSD, INTEGER
//          A code indicating how to set the random number seed.
//          = 0:  Set the seed to a default value before each run
//          = 1:  Initialize the seed to a default value only before the
//                first run
//          = 2:  Like 1, but use the seed values on the next line
//
// If line 8 was 2:
//
// line 9:  INTEGER array, dimension (4)
//          Four integer values for the random number seed.
//
// lines 9-EOF:  Lines specifying matrix types, as for NEP.
//          The 3-character path name is 'GSV' for the generalized
//          SVD routines.
//
//-----------------------------------------------------------------------
//
// NMAX is currently set to 132 and must be at least 12 for some of the
// precomputed examples, and LWORK = NMAX*(5*NMAX+20) in the parameter
// statements below.  For SVD, we assume NRHS may be as big as N.  The
// parameter NEED is set to 14 to allow for 14 N-by-N matrices for ZGG.
func TestZeig(t *testing.T) {
	var csd, fatal, glm, gqr, gsv, lse, nep, sep, svd, tstchk, tstdif, tstdrv, tsterr, zbb, zbk, zbl, zes, zev, zgg, zgk, zgl, zgs, zgv, zgx, zhb, zsx, zvx, zxv bool
	var eps, thresh, thrshn float64
	var i, info, k, liwork, lwork, maxtyp, ncmax, need, newsd, nk, nmax, nn, nout, nparms, nrhs, ntypes, versMajor, versMinor, versPatch int

	nmax = 132
	ncmax = 20
	need = 14
	lwork = nmax * (5*nmax + 20)
	liwork = nmax * (nmax + 20)
	nout = 6
	dotype := make([]bool, 30)
	logwrk := make([]bool, 132)
	iacc22 := make([]int, 20)
	inibl := make([]int, 20)
	inmin := make([]int, 20)
	inwin := make([]int, 20)
	iseed := make([]int, 4)
	ishfts := make([]int, 20)
	iwork := make([]int, liwork)
	kval := make([]int, 20)
	mval := make([]int, 20)
	mxbval := make([]int, 20)
	nbcol := make([]int, 20)
	nbmin := make([]int, 20)
	nbval := make([]int, 20)
	nsval := make([]int, 20)
	nval := make([]int, 20)
	nxval := make([]int, 20)
	pval := make([]int, 20)
	iparms := &gltest.Common.Claenv.Iparms
	selval := &gltest.Common.Sslct.Selval
	selwi := &gltest.Common.Sslct.Selwi
	selwr := &gltest.Common.Sslct.Selwr
	*iparms = make([]int, 100)
	*selval = make([]bool, 20)
	*selwi = vf(20)
	*selwr = vf(20)
	taua := cvf(nmax)
	taub := cvf(nmax)
	work := cvf(lwork)
	x := cvf(5 * nmax)
	alpha := vf(132)
	beta := vf(132)
	result := vf(500)
	rwork := vf(lwork)
	s := vf(nmax * nmax)
	a := func() []*mat.CMatrix {
		arr := make([]*mat.CMatrix, need)
		for u := 0; u < need; u++ {
			arr[u] = cmf(nmax, nmax, opts)
		}
		return arr
	}()
	b := func() []*mat.CMatrix {
		arr := make([]*mat.CMatrix, 5)
		for u := 0; u < 5; u++ {
			arr[u] = cmf(nmax, nmax, opts)
		}
		return arr
	}()
	c := cmf(ncmax*ncmax, ncmax*ncmax, opts)
	dc := func() []*mat.CVector {
		arr := make([]*mat.CVector, 6)
		for u := 0; u < 6; u++ {
			arr[u] = cvf(nmax)
		}
		return arr
	}()
	dr := func() []*mat.Vector {
		arr := make([]*mat.Vector, 12)
		for u := 0; u < 12; u++ {
			arr[u] = vf(lwork)
		}
		return arr
	}()

	ioldsd := []int{0, 0, 0, 1}

	fatal = false

	golapack.Ilaver(&versMajor, &versMinor, &versPatch)
	fmt.Printf("\n LAPACK VERSION %1d.%1d.%1d\n", versMajor, versMinor, versPatch)
	fmt.Printf("\n The following parameter values will be used:\n")

	//     Calculate and print the machine dependent constants.
	fmt.Printf("\n")
	eps = golapack.Dlamch(Underflow)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "underflow", eps)
	eps = golapack.Dlamch(Overflow)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "overflow ", eps)
	eps = golapack.Dlamch(Epsilon)
	fmt.Printf(" Relative machine %s is taken to be%16.6E\n", "precision", eps)

	//     Read the first line and set the 3-character test path
	// []string{"NEP", "SEP", "SE2", "SVD", "ZEC", "ZES", "ZEV", "ZSX", "ZVX", "ZGG", "ZGV", "ZGS", "ZGX", "ZXV", "ZHB", "ZSG", "ZBL", "ZBK", "ZGL", "ZGK", "ZBB", "GLM", "GQR", "GSV", "CSD", "LSE"}
	for _, path := range []string{"NEP", "SEP", "SE2", "SVD", "ZEC", "ZES", "ZEV", "ZSX", "ZVX", "ZGG", "ZGV", "ZGS", "ZGX", "ZXV", "ZHB", "ZSG", "ZBL", "ZBK", "ZGL", "ZGK", "ZBB", "GLM", "GQR", "GSV", "CSD", "LSE"} {
		nep = string(path) == "NEP" || string(path) == "ZHS"
		sep = string(path) == "SEP" || string(path) == "ZST" || string(path) == "ZSG" || string(path) == "SE2"
		svd = string(path) == "SVD" || string(path) == "ZBD"
		zev = string(path) == "ZEV"
		zes = string(path) == "ZES"
		zvx = string(path) == "ZVX"
		zsx = string(path) == "ZSX"
		zgg = string(path) == "ZGG"
		zgs = string(path) == "ZGS"
		zgx = string(path) == "ZGX"
		zgv = string(path) == "ZGV"
		zxv = string(path) == "ZXV"
		zhb = string(path) == "ZHB"
		zbb = string(path) == "ZBB"
		glm = string(path) == "GLM"
		gqr = string(path) == "GQR" || string(path) == "GRQ"
		gsv = string(path) == "GSV"
		csd = string(path) == "CSD"
		lse = string(path) == "LSE"
		zbl = string(path) == "ZBL"
		zbk = string(path) == "ZBK"
		zgl = string(path) == "ZGL"
		zgk = string(path) == "ZGK"

		//     Report values of parameters.
		fmt.Printf("\n------------------------------------------------------\n")
		if nep {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem routines\n")
		} else if sep {
			fmt.Printf(" Tests of the Hermitian Eigenvalue Problem routines\n")
		} else if svd {
			fmt.Printf(" Tests of the Singular Value Decomposition routines\n")
		} else if zev {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Driver\n    ZGEEV (eigenvalues and eigevectors)\n")
		} else if zes {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Driver\n    ZGEES (Schur form)\n")
		} else if zvx {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    ZGEEVX (eigenvalues, eigenvectors and condition numbers)\n")
		} else if zsx {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    ZGEESX (Schur form and condition numbers)\n")
		} else if zgg {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem routines\n")
		} else if zgs {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver ZGGES\n")
		} else if zgx {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver ZGGESX\n")
		} else if zgv {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver ZGGEV\n")
		} else if zxv {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver ZGGEVX\n")
		} else if zhb {
			fmt.Printf(" Tests of ZHBTRD\n (reduction of a Hermitian band matrix to real tridiagonal form)\n")
		} else if zbb {
			fmt.Printf(" Tests of ZGBBRD\n (reduction of a general band matrix to real bidiagonal form)\n")
		} else if glm {
			fmt.Printf(" Tests of the Generalized Linear Regression Model routines\n")
		} else if gqr {
			fmt.Printf(" Tests of the Generalized QR and RQ routines\n")
		} else if gsv {
			fmt.Printf(" Tests of the Generalized Singular Value Decomposition routines\n")
		} else if csd {
			fmt.Printf(" Tests of the CS Decomposition routines\n")
		} else if lse {
			fmt.Printf(" Tests of the Linear Least Squares routines\n")
		} else if zbl {
			//        ZGEBAL:  Balancing
			Zchkbl(t)
			continue
		} else if zbk {
			//        ZGEBAK:  Back transformation
			Zchkbk(t)
			continue
		} else if zgl {
			//        ZGGBAL:  Balancing
			Zchkgl(t)
			continue
		} else if zgk {
			//        ZGGBAK:  Back transformation
			Zchkgk(t)
			continue
		} else if string(path) == "ZEC" {
			//        ZEC:  Eigencondition estimation
			thresh = 20.0
			Xlaenv(1, 1)
			Xlaenv(12, 1)
			tsterr = true
			Zchkec(&thresh, &tsterr, t)
			continue
		} else {
			fmt.Printf(" %3s:  Unrecognized path name\n", path)
			continue
		}

		for i = 1; i <= 4; i++ {
			iseed[i-1] = ioldsd[i-1]
		}

		if fatal {
			fmt.Printf("\n Execution not attempted due to input errors\n")
			panic("")
		}

		if path == "ZHS" || path == "NEP" {
			//        -------------------------------------
			//        NEP:  Nonsymmetric Eigenvalue Problem
			//        -------------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			//           NS    = number of shifts
			//           MAXB  = minimum submatrix size
			// NEP
			nval = []int{0, 1, 2, 3, 5, 10, 16}
			nbval = []int{1, 3, 3, 3, 20}
			nbmin = []int{2, 2, 2, 2, 2}
			nxval = []int{1, 0, 5, 9, 1}
			inmin = []int{11, 12, 11, 15, 11}
			inwin = []int{2, 3, 5, 3, 2}
			inibl = []int{0, 5, 7, 3, 200}
			ishfts = []int{1, 2, 4, 2, 1}
			iacc22 = []int{0, 1, 2, 0, 1}
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 21
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Alareq(&ntypes, &dotype)
			Xlaenv(1, 1)
			if tsterr {
				Zerrhs([]byte("ZHSEQR"), t)
			}
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])
				Xlaenv(12, maxint(11, inmin[i-1]))
				Xlaenv(13, inwin[i-1])
				Xlaenv(14, inibl[i-1])
				Xlaenv(15, ishfts[i-1])
				Xlaenv(16, iacc22[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NX =%4d, INMIN=%4d, INWIN =%4d, INIBL =%4d, ISHFTS =%4d, IACC22 =%4d\n", path, nbval[i-1], nbmin[i-1], nxval[i-1], maxint(11, inmin[i-1]), inwin[i-1], inibl[i-1], ishfts[i-1], iacc22[i-1])
				Zchkhs(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[4], &nmax, a[5], a[6], dc[0], dc[1], a[7], a[8], a[9], a[10], a[11], dc[2], work, &lwork, rwork, &iwork, &logwrk, result, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZCHKHS", info)
				}
			}

		} else if path == "ZST" || path == "SEP" || path == "SE2" {
			//        ----------------------------------
			//        SEP:  Symmetric Eigenvalue Problem
			//        ----------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			nval = []int{0, 1, 2, 3, 5, 20}
			nbval = []int{1, 3, 3, 3, 10}
			nbmin = []int{2, 2, 2, 2, 2}
			nxval = []int{1, 0, 5, 9, 1}
			thresh = 50.0
			tstchk = true
			tstdrv = true
			tsterr = true
			newsd = 1
			ntypes = 21
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			maxtyp = 21
			ntypes = minint(maxtyp, ntypes)
			Alareq(&ntypes, &dotype)
			dotype[8] = false
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			Xlaenv(9, 25)
			if tsterr {
				Zerrst([]byte("ZST"), t)
			}
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NX =%4d\n", path, nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					if path == "SE2" {
						Zchkst2stg(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1].CVector(0, 0), dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], dr[6], dr[7], dr[8], dr[9], dr[10], a[2], &nmax, a[3], a[4].CVector(0, 0), dc[0], a[5], work, &lwork, rwork, &lwork, &iwork, &liwork, result, &info, t)
					} else {
						Zchkst(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1].CVector(0, 0), dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], dr[6], dr[7], dr[8], dr[9], dr[10], a[2], &nmax, a[3], a[4].CVector(0, 0), dc[0], a[5], work, &lwork, rwork, &lwork, &iwork, &liwork, result, &info, t)
					}
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "ZCHKST", info)
					}
				}
				if tstdrv {
					if path == "SE2" {
						Zdrvst2stg(&nn, &nval, func() *int { y := 18; return &y }(), &dotype, &iseed, &thresh, &nout, a[0], &nmax, dr[2], dr[3], dr[4], dr[7], dr[8], dr[9], a[1], &nmax, a[2], dc[0], a[3], work, &lwork, rwork, &lwork, &iwork, &liwork, result, &info, t)
					} else {
						Zdrvst(&nn, &nval, func() *int { y := 18; return &y }(), &dotype, &iseed, &thresh, &nout, a[0], &nmax, dr[2], dr[3], dr[4], dr[7], dr[8], dr[9], a[1], &nmax, a[2], dc[0], a[3], work, &lwork, rwork, &lwork, &iwork, &liwork, result, &info, t)
					}
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "ZDRVST", info)
					}
				}
			}

		} else if path == "ZSG" {
			//        ----------------------------------------------
			//        ZSG:  Hermitian Generalized Eigenvalue Problem
			//        ----------------------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			nval = []int{0, 1, 2, 3, 5, 10, 16}
			nbval = []int{1, 3, 20}
			nbmin = []int{2, 2, 2}
			nxval = []int{1, 1, 1}
			thresh = 20.0
			tstchk = true
			tstdrv = true
			tsterr = true
			newsd = 1
			ntypes = 21
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Alareq(&ntypes, &dotype)
			Xlaenv(9, 25)
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NX =%4d\n", path, nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					Zdrvsg2stg(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], &nmax, dr[2], dr[3], a[2], &nmax, a[3], a[4], a[5].CVector(0, 0), a[6].CVector(0, 0), work, &lwork, rwork, &lwork, &iwork, &liwork, result, &info, t)
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "ZDRVSG", info)
					}
				}
			}

		} else if path == "ZBD" || path == "SVD" {
			//        ----------------------------------
			//        SVD:  Singular Value Decomposition
			//        ----------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			//           NRHS  = number of right hand sides
			mval = []int{0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 10, 10, 16, 16, 30, 30, 40, 40}
			nval = []int{0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 3, 10, 16, 10, 16, 30, 40, 30, 40}
			nbval = []int{1, 3, 3, 3, 20}
			nbmin = []int{2, 2, 2, 2, 2}
			nxval = []int{1, 0, 5, 9, 1}
			nsval = []int{2, 0, 2, 2, 2}
			thresh = 50.0
			tstchk = true
			tstdrv = true
			tsterr = true
			newsd = 1
			ntypes = 16
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Alareq(&ntypes, &dotype)
			Xlaenv(9, 25)

			//        Test the error exits
			Xlaenv(1, 1)
			if tsterr && tstchk {
				Zerrbd([]byte("ZBD"), t)
			}
			if tsterr && tstdrv {
				Zerred([]byte("ZBD"), t)
			}

			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(3, nxval[i-1])
				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NX =%4d, NRHS =%4d\n", path, nbval[i-1], nbmin[i-1], nxval[i-1], nrhs)
				if tstchk {
					Zchkbd(&nn, &mval, &nval, &maxtyp, &dotype, &nrhs, &iseed, &thresh, a[0], &nmax, dr[0], dr[1], dr[2], dr[3], a[1], &nmax, a[2], a[3], a[4], &nmax, a[5], &nmax, a[6], a[7], work, &lwork, rwork, &nout, &info, t)
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "ZCHKBD", info)
					}
				}
				if tstdrv {
					Zdrvbd(&nn, &mval, &nval, &maxtyp, &dotype, &iseed, &thresh, a[0], &nmax, a[1], &nmax, a[2], &nmax, a[3], a[4], a[5], dr[0], dr[1], dr[2], work, &lwork, rwork, &iwork, &nout, &info, t)
				}
			}

		} else if path == "ZEV" {
			//        --------------------------------------------
			//        ZEV:  Nonsymmetric Eigenvalue Problem Driver
			//              ZGEEV (eigenvalues and eigenvectors)
			//        --------------------------------------------
			nval = []int{0, 1, 2, 3, 5, 10, 20}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			thresh = 20.0
			tsterr = true
			newsd = 2
			iseed = []int{2518, 3899, 995, 397}
			ntypes = 21
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if ntypes <= 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerred([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Zdrvev(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], dc[0], dc[1], a[2], &nmax, a[3], &nmax, a[4], &nmax, result, work, &lwork, rwork, &iwork, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZGEEV", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "ZES" {
			//        --------------------------------------------
			//        ZES:  Nonsymmetric Eigenvalue Problem Driver
			//              ZGEES (Schur form)
			//        --------------------------------------------
			nval = []int{0, 1, 2, 3, 5, 10, 20}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			thresh = 20.0
			tsterr = true
			newsd = 2
			iseed = []int{2518, 3899, 995, 397}
			ntypes = 21
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if ntypes <= 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerred([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Zdrves(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], dc[0], dc[1], a[3], &nmax, result, work, &lwork, rwork, &iwork, &logwrk, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZGEES", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "ZVX" {
			//        --------------------------------------------------------------
			//        ZVX:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              ZGEEVX (eigenvalues, eigenvectors and condition numbers)
			//        --------------------------------------------------------------
			nval = []int{0, 1, 2, 3, 5, 10, 20}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			thresh = 20.0
			tsterr = true
			newsd = 2
			iseed = []int{2518, 3899, 995, 397}
			ntypes = 21
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if ntypes < 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerred([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Zdrvvx(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], dc[0], dc[1], a[2], &nmax, a[3], &nmax, a[4], &nmax, dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], dr[6], dr[7], result, work, &lwork, rwork, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZGEEVX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "ZSX" {
			//        ---------------------------------------------------
			//        ZSX:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              ZGEESX (Schur form and condition numbers)
			//        ---------------------------------------------------
			nval = []int{0, 1, 2, 3, 5, 10, 20}
			nbval = []int{3}
			nbmin = []int{3}
			nxval = []int{1}
			inmin = []int{11}
			inwin = []int{4}
			inibl = []int{8}
			ishfts = []int{2}
			iacc22 = []int{0}
			thresh = 20.0
			tsterr = true
			newsd = 2
			iseed = []int{2518, 3899, 995, 397}
			ntypes = 21
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if ntypes < 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerred([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Zdrvsx(&nn, &nval, &ntypes, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], dc[0], dc[1], dc[2], a[3], &nmax, a[4], result, work, &lwork, rwork, &logwrk, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZGEESX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "ZGG" {
			//        -------------------------------------------------
			//        ZGG:  Generalized Nonsymmetric Eigenvalue Problem
			//        -------------------------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NS    = number of shifts
			//           MAXB  = minimum submatrix size
			//           IACC22: structured matrix multiply
			//           NBCOL = minimum column dimension for blocks
			nval = []int{0, 1, 2, 3, 5, 10, 16}
			nbval = []int{1, 1, 2, 2}
			nbmin = []int{40, 40, 2, 2}
			nsval = []int{2, 4, 2, 4}
			mxbval = []int{40, 40, 2, 2}
			iacc22 = []int{1, 2, 1, 2}
			nbcol = []int{40, 40, 2, 2}
			thresh = 20.0
			tstchk = true
			tstdrv = false
			tsterr = true
			newsd = 1
			ntypes = 26
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Alareq(&ntypes, &dotype)
			Xlaenv(1, 1)
			if tstchk && tsterr {
				Zerrgg([]byte(path), t)
			}
			for i = 1; i <= nparms; i++ {
				Xlaenv(1, nbval[i-1])
				Xlaenv(2, nbmin[i-1])
				Xlaenv(4, nsval[i-1])
				Xlaenv(8, mxbval[i-1])
				Xlaenv(16, iacc22[i-1])
				Xlaenv(5, nbcol[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NS =%4d, MAXB =%4d, IACC22 =%4d, NBCOL =%4d\n", path, nbval[i-1], nbmin[i-1], nsval[i-1], mxbval[i-1], iacc22[i-1], nbcol[i-1])
				tstdif = false
				thrshn = 10.
				if tstchk {
					Zchkgg(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &tstdif, &thrshn, &nout, a[0], &nmax, a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], &nmax, a[9], a[10], a[11], dc[0], dc[1], dc[2], dc[3], a[12], a[13], work, &lwork, rwork, &logwrk, result, &info, t)
					if info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "ZCHKGG", info)
					}
				}
			}

		} else if path == "ZGS" {
			//        -------------------------------------------------
			//        ZGS:  Generalized Nonsymmetric Eigenvalue Problem
			//              ZGGES (Schur form)
			//        -------------------------------------------------
			nval = []int{2, 6, 10, 12, 20, 30}
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			nbcol = []int{1}
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 26
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if newsd == 0 {
				for k = 1; k <= 4; k++ {
					iseed[k-1] = ioldsd[k-1]
				}
			}
			if ntypes <= 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerrgg([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Zdrges(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], dc[0], dc[1], work, &lwork, rwork, result, &logwrk, &info, t)

				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGES", info)
				}

				// Blocked version
				Zdrges3(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], dc[0], dc[1], work, &lwork, rwork, result, &logwrk, &info, t)

				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGES3", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if zgx {
			//        -------------------------------------------------
			//        ZGX  Generalized Nonsymmetric Eigenvalue Problem
			//              ZGGESX (Schur form and condition numbers)
			//        -------------------------------------------------
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			nbcol = []int{1}
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 5
			maxtyp = ntypes
			nn = 2
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if newsd == 0 {
				for k = 1; k <= 4; k++ {
					iseed[k-1] = ioldsd[k-1]
				}
			}
			if nn < 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerrgg([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Xlaenv(5, 2)
				Zdrgsx(&nn, &ncmax, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[4], a[5], dc[0], dc[1], c, toPtr(ncmax*ncmax), s, work, &lwork, rwork, &iwork, &liwork, &logwrk, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGSX", info)
				}

				nn = 0
				Zdrgsx(&nn, &ncmax, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[4], a[5], dc[0], dc[1], c, toPtr(ncmax*ncmax), s, work, &lwork, rwork, &iwork, &liwork, &logwrk, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGSX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "ZGV" {
			//        -------------------------------------------------
			//        ZGV:  Generalized Nonsymmetric Eigenvalue Problem
			//              ZGGEV (Eigenvalue/vector form)
			//        -------------------------------------------------
			nval = []int{2, 6, 8, 10, 12, 20}
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			nbcol = []int{1}
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 26
			maxtyp = ntypes
			nn = len(nval)
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if ntypes <= 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerrgg([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Zdrgev(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], a[8], &nmax, dc[0], dc[1], dc[2], dc[3], work, &lwork, rwork, result, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGEV", info)
				}

				// Blocked version
				Xlaenv(16, 2)
				Zdrgev3(&nn, &nval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], a[6], &nmax, a[7], a[8], &nmax, dc[0], dc[1], dc[2], dc[3], work, &lwork, rwork, result, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGEV3", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if zxv {
			//        -------------------------------------------------
			//        ZXV:  Generalized Nonsymmetric Eigenvalue Problem
			//              ZGGEVX (eigenvalue/vector with condition numbers)
			//        -------------------------------------------------
			nbval = []int{1}
			nbmin = []int{1}
			nxval = []int{1}
			nsval = []int{2}
			nbcol = []int{1}
			thresh = 10.0
			tsterr = true
			newsd = 0
			ntypes = 2
			maxtyp = ntypes
			nn = 6
			nparms = len(nbval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if newsd == 0 {
				for k = 1; k <= 4; k++ {
					iseed[k-1] = ioldsd[k-1]
				}
			}
			if nn < 0 {
				fmt.Printf(" %3s routines were not tested\n", path)
			} else {
				if tsterr {
					Zerrgg([]byte(path), t)
				}
				Alareq(&ntypes, &dotype)
				Zdrgvx(&nn, &thresh, &nout, a[0], &nmax, a[1], a[2], a[3], dc[0], dc[1], a[4], a[5], &iwork[0], &iwork[1], dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], work, &lwork, rwork, toSlice(&iwork, 2), toPtr(liwork-2), result, &logwrk, &info, t)

				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGVX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "ZHB" {
			//        ------------------------------
			//        ZHB:  Hermitian Band Reduction
			//        ------------------------------
			nval = []int{5, 20}
			kval = []int{0, 1, 2, 5, 16}
			thresh = 20.0
			tsterr = true
			newsd = 0
			ntypes = 15
			maxtyp = ntypes
			nn = len(nval)
			nk = len(kval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Alareq(&ntypes, &dotype)
			if newsd == 0 {
				for k = 1; k <= 4; k++ {
					iseed[k-1] = ioldsd[k-1]
				}
			}
			if tsterr {
				Zerrst([]byte("ZHB"), t)
			}
			Zchkhb2stg(&nn, &nval, &nk, &kval, &maxtyp, &dotype, &iseed, &thresh, &nout, a[0], &nmax, dr[0], dr[1], dr[2], dr[3], dr[4], a[1], &nmax, work, &lwork, rwork, result, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "ZCHKHB", info)
			}

		} else if path == "ZBB" {
			//        ------------------------------
			//        ZBB:  General Band Reduction
			//        ------------------------------
			mval = []int{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 16, 16}
			nval = []int{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 10, 16, 10, 16}
			kval = []int{0, 1, 2, 3, 16}
			nsval = []int{1, 2}
			thresh = 20.0
			tsterr = false
			newsd = 1
			ntypes = 15
			maxtyp = ntypes
			nn = len(nval)
			nk = len(kval)
			nparms = len(nsval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Alareq(&ntypes, &dotype)
			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NRHS =%4d\n", path, nrhs)
				Zchkbb(&nn, &mval, &nval, &nk, &kval, &maxtyp, &dotype, &nrhs, &iseed, &thresh, &nout, a[0], &nmax, a[1].Off(0, 0).UpdateRows(2*nmax), toPtr(2*nmax), dr[0], dr[1], a[3], &nmax, a[4], &nmax, a[5], &nmax, a[6], work, &lwork, rwork, result, &info, t)
				if info != 0 {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZCHKBB", info)
				}
			}

		} else if path == "GLM" {
			//        -----------------------------------------
			//        GLM:  Generalized Linear Regression Model
			//        -----------------------------------------
			mval = []int{0, 5, 8, 15, 20, 40}
			pval = []int{9, 0, 15, 12, 15, 30}
			nval = []int{5, 5, 10, 25, 30, 40}
			nsval = []int{1, 2}
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			maxtyp = ntypes
			nn = len(nval)
			nk = len(kval)
			nparms = len(nsval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Zerrgg([]byte("GLM"), t)
			}
			Zckglm(&nn, &nval, &mval, &pval, &ntypes, &iseed, &thresh, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), x, work, dr[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "ZCKGLM", info)
			}

		} else if path == "GQR" {
			//        ------------------------------------------
			//        GQR:  Generalized QR and RQ factorizations
			//        ------------------------------------------
			mval = []int{0, 3, 10}
			pval = []int{0, 5, 20}
			nval = []int{0, 3, 30}
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			maxtyp = ntypes
			nn = len(nval)
			nk = len(kval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Zerrgg([]byte("GQR"), t)
			}
			Zckgqr(&nn, &mval, &nn, &pval, &nn, &nval, &ntypes, &iseed, &thresh, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), taua, b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), b[4].CVector(0, 0), taub, work, dr[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "ZCKGQR", info)
			}

		} else if path == "GSV" {
			//        ----------------------------------------------
			//        GSV:  Generalized Singular Value Decomposition
			//        ----------------------------------------------
			mval = []int{0, 5, 9, 10, 20, 12, 12, 40}
			pval = []int{4, 0, 12, 14, 10, 10, 20, 15}
			nval = []int{3, 10, 15, 12, 8, 20, 8, 20}
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			maxtyp = ntypes
			nn = len(nval)
			nk = len(kval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Zerrgg([]byte("GSV"), t)
			}
			Zckgsv(&nn, &mval, &pval, &nval, &ntypes, &iseed, &thresh, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), a[2].CVector(0, 0), b[2].CVector(0, 0), a[3].CVector(0, 0), alpha, beta, b[3].CVector(0, 0), &iwork, work, dr[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "ZCKGSV", info)
			}

		} else if path == "CSD" {
			//        ----------------------------------------------
			//        CSD:  CS Decomposition
			//        ----------------------------------------------
			mval = []int{0, 10, 10, 10, 10, 21, 24, 30, 22, 32, 55}
			pval = []int{0, 4, 4, 0, 10, 9, 10, 20, 12, 12, 40}
			nval = []int{0, 0, 10, 4, 4, 15, 12, 8, 20, 8, 20}
			thresh = 30.0
			tsterr = true
			newsd = 1
			ntypes = 4
			maxtyp = ntypes
			nn = len(nval)
			nk = len(kval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Zerrgg([]byte("CSD"), t)
			}
			Zckcsd(&nn, &mval, &pval, &nval, &ntypes, &iseed, &thresh, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), a[5].CVector(0, 0), rwork, &iwork, work, dr[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "ZCKCSD", info)
			}

		} else if path == "LSE" {
			//        --------------------------------------
			//        LSE:  Constrained Linear Least Squares
			//        --------------------------------------
			mval = []int{6, 0, 5, 8, 10, 30}
			pval = []int{0, 5, 5, 5, 8, 20}
			nval = []int{5, 5, 6, 8, 12, 40}
			thresh = 20.0
			tsterr = true
			newsd = 1
			ntypes = 8
			maxtyp = ntypes
			nn = len(nval)
			nk = len(kval)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			Xlaenv(1, 1)
			if tsterr {
				Zerrgg([]byte("LSE"), t)
			}
			Zcklse(&nn, &mval, &pval, &nval, &ntypes, &iseed, &thresh, &nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), x, work, dr[0], &nout, &info, t)
			if info != 0 {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "ZCKLSE", info)
			}

		} else {
			fmt.Printf("\n")
			fmt.Printf(" %3s:  Unrecognized path name\n", path)
		}
	}

	fmt.Printf("\n\n End of tests\n")
}
