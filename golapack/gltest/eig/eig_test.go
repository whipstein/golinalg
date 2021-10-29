package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
	"github.com/whipstein/golinalg/util"
)

// Dchkee tests the DOUBLE PRECISION LAPACK subroutines for the matrix
// eigenvalue problem.  The test paths in this version are
//
// Nep (Nonsymmetric Eigenvalue Problem):
//     Test DGEHRD, DORGHR, Dhseqr, DTREVC, DHSEIN, and DORMHR
//
// Sep (Symmetric Eigenvalue Problem):
//     Test DSYTRD, DORGTR, DSTEQR, DSTERF, DSTEIN, DSTEDC,
//     and drivers DSYEV(X), DSBEV(X), DSPEV(X), DSTEV(X),
//                 DSYEVD,   DSBEVD,   DSPEVD,   DSTEVD
//
// Svd (Singular Value Decomposition):
//     Test DGEBRD, DORGBR, DBDSQR, DBDSDC
//     and the drivers DGESVD, DGESDD
//
// Dev (Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test dgeev
//
// Des (Nonsymmetric Schur form Driver):
//     Test dgees
//
// Dvx (Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test dgeevx
//
// Dsx (Nonsymmetric Schur form Expert Driver):
//     Test dgeesx
//
// Dgg (Generalized Nonsymmetric Eigenvalue Problem):
//     Test DGGHD3, DGGBAL, DGGBAK, DHGEQZ, and DTGEVC
//
// Dgs (Generalized Nonsymmetric Schur form Driver):
//     Test DGGES
//
// Dgv (Generalized Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test DGGEV
//
// Dgx (Generalized Nonsymmetric Schur form Expert Driver):
//     Test DGGESX
//
// Dxv (Generalized Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test DGGEVX
//
// Dsg (Symmetric Generalized Eigenvalue Problem):
//     Test DSYGST, DSYGV, DSYGVD, DSYGVX, DSPGST, DSPGV, DSPGVD,
//     DSPGVX, DSBGST, DSBGV, DSBGVD, and DSBGVX
//
// Dsb (Symmetric Band Eigenvalue Problem):
//     Test DSBTRD
//
// Dbb (Band Singular Value Decomposition):
//     Test DGBBRD
//
// Dec (Eigencondition estimation):
//     Test DLALN2, DLASY2, DLAEQU, DLAEXC, DTRSYL, DTREXC, DTRSNA,
//     DTRSEN, and DLAQTR
//
// Dbl (Balancing a general matrix)
//     Test DGEBAL
//
// Dbk (Back transformation on a balanced matrix)
//     Test DGEBAK
//
// Dgl (Balancing a matrix pair)
//     Test DGGBAL
//
// Dgk (Back transformation on a matrix pair)
//     Test DGGBAK
//
// Glm (Generalized Linear Regression Model):
//     Tests DGGGLM
//
// Gqr (Generalized QR and RQ factorizations):
//     Tests DGGQRF and DGGRQF
//
// Gsv (Generalized Singular Value Decomposition):
//     Tests DGGSVD, DGGSVP, DTGSJA, DLAGS2, DLAPLL, and DLAPMT
//
// Csd (CS decomposition):
//     Tests DORCSD
//
// Lse (Constrained Linear Least Squares):
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
	var err error
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

	for _, path := range []string{"Nep", "Sep", "Se2", "Svd", "Dec", "Dev", "Des", "Dvx", "Dsx", "Dgg", "Dgs", "Dgv", "Dgx", "Dxv", "Dsb", "Dsg", "Dbl", "Dbk", "Dgl", "Dgk", "Dbb", "Glm", "Gqr", "Gsv", "Csd", "Lse"} {
		c3 := path
		tstchk = false
		tstdrv = false
		tsterr = false
		dgx := path == "Dgx"
		dxv := path == "Dxv"

		versMajor, versMinor, versPatch = golapack.Ilaver()
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

		if c3 == "Dhs" || c3 == "Nep" {
			//        -------------------------------------
			//        Nep:  Nonsymmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			if tsterr {
				derrhs("Dhseqr", t)
			}
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])
				xlaenv(12, max(11, inmin[i-1]))
				xlaenv(13, inwin[i-1])
				xlaenv(14, inibl[i-1])
				xlaenv(15, ishfts[i-1])
				xlaenv(16, iacc22[i-1])
				//
				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NX =%4d, INMIN=%4d, INWIN =%4d, INIBL =%4d, ISHFTS =%4d, IACC22 =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1], max(11, inmin[i-1]), inwin[i-1], inibl[i-1], ishfts[i-1], iacc22[i-1])
				if err = dchkhs(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1], a[2], a[3], a[4], a[5], a[6], d[0], d[1], d[2], d[3], d[4], d[5], a[7], a[8], a[9], a[10], a[11], d[6], work, lwork, iwork, logwrk, result, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "dchkhs", info)
				}
			}

		} else if c3 == "Dst" || c3 == "Sep" || c3 == "Se2" {
			//        ----------------------------------
			//        Sep:  Symmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			dotype[8] = false
			ntypes = min(maxtyp, ntypes)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			xlaenv(9, 25)
			if tsterr {
				derrst("Dst", t)
			}
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NB =%4d, NBMIN =%4d, NX =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					if c3 == "Se2" {
						dotype[8] = false
						err = dchkst2stg(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1].VectorIdx(0), d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], a[2], a[3], a[4].VectorIdx(0), d[11], a[5], work, lwork, iwork, liwork, result)
					} else {
						err = dchkst(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1].VectorIdx(0), d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], a[2], a[3], a[4].VectorIdx(0), d[11], a[5], work, lwork, iwork, liwork, result, t)
					}
					if err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "dchkst", info)
					}
				}
				if tstdrv {
					if c3 == "Se2" {
						dotype[8] = false
						err = ddrvst2stg(nn, nval, 18, dotype, iseed, thresh, nout, a[0], d[2], d[3], d[4], d[5], d[7], d[8], d[9], d[10], a[1], a[2], d[11], a[3], work, lwork, iwork, liwork, result, t)
					} else {
						err = ddrvst(nn, nval, 18, dotype, iseed, thresh, nout, a[0], d[2], d[3], d[4], d[5], d[7], d[8], d[9], d[10], a[1], a[2], d[11], a[3], work, lwork, iwork, liwork, result, t)
					}
					if err != nil || info != 0 {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "ddrvst", info)
					}
				}
			}

		} else if c3 == "Dsg" {
			//        ----------------------------------------------
			//        Dsg:  Symmetric Generalized Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(9, 25)
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NB =%4d, NBMIN =%4d, NX =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					//               CALL ddrvsg( NN, NVAL, MAXTYP, DOTYPE, ISEED, THRESH,
					//     $                      NOUT, A( 1, 1 ), NMAX, A( 1, 2 ), NMAX,
					//     $                      D( 1, 3 ), A( 1, 3 ), NMAX, A( 1, 4 ),
					//     $                      A( 1, 5 ), A( 1, 6 ), A( 1, 7 ), WORK,
					//     $                      LWORK, IWORK, LIWORK, RESULT, INFO )
					if err = ddrvsg2stg(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1], d[2], d[2], a[2], a[3], a[4], a[5].VectorIdx(0), a[6].VectorIdx(0), work, lwork, iwork, liwork, result, t); err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "ddrvsg", info)
					}
				}
			}

		} else if c3 == "Dbd" || c3 == "Svd" {
			//        ----------------------------------
			//        Svd:  Singular Value Decomposition
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			xlaenv(9, 25)

			//        Test the error exits
			if tsterr && tstchk {
				derrbd("Dbd", t)
			}
			if tsterr && tstdrv {
				derred("Dbd", t)
			}

			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])
				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NB =%4d, NBMIN =%4d, NX =%4d, NRHS =%4d\n", c3, nbval[i-1], nbmin[i-1], nxval[i-1], nrhs)
				if tstchk {
					if err = dchkbd(nn, mval, nval, maxtyp, dotype, nrhs, &iseed, thresh, a[0], d[0], d[1], d[2], d[3], a[1], a[2], a[3], a[4], a[5], a[6], a[7], work, lwork, iwork, nout, t); err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %v\n", "dchkbd", err)
					}
				}
				if tstdrv {
					if err = ddrvbd(nn, mval, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], a[2], a[3], a[4], a[5], d[0], d[1], d[2], work, lwork, iwork, nout, t); err != nil {
						t.Fail()
					}
				}
			}

		} else if c3 == "Dev" {
			//        --------------------------------------------
			//        Dev:  Nonsymmetric Eigenvalue Problem Driver
			//              dgeev (eigenvalues and eigenvectors)
			//        --------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Driver\n    dgeev (eigenvalues and eigevectors)\n")
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(12, max(11, inmin[0]))
			xlaenv(13, inwin[0])
			xlaenv(14, inibl[0])
			xlaenv(15, ishfts[0])
			xlaenv(16, iacc22[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derred(c3, t)
				}
				if err = ddrvev(nn, nval, ntypes, dotype, iseed, thresh, nout, a[0], a[1], d[0], d[1], d[2], d[3], a[2], a[3], a[4], result, work, lwork, iwork, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "dgeev", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if c3 == "Des" {
			//        --------------------------------------------
			//        Des:  Nonsymmetric Eigenvalue Problem Driver
			//              dgees (Schur form)
			//        --------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Driver\n    dgees (Schur form)\n")
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(12, max(11, inmin[0]))
			xlaenv(13, inwin[0])
			xlaenv(14, inibl[0])
			xlaenv(15, ishfts[0])
			xlaenv(16, iacc22[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derred(c3, t)
				}
				if err = ddrves(nn, nval, ntypes, dotype, iseed, thresh, nout, a[0], a[1], a[2], d[0], d[1], d[2], d[3], a[3], result, work, lwork, iwork, logwrk, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "dgees", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if c3 == "Dvx" {
			//        --------------------------------------------------------------
			//        Dvx:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              dgeevx (eigenvalues, eigenvectors and condition numbers)
			//        --------------------------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    dgeevx (eigenvalues, eigenvectors and condition numbers)\n")
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(12, max(11, inmin[0]))
			xlaenv(13, inwin[0])
			xlaenv(14, inibl[0])
			xlaenv(15, ishfts[0])
			xlaenv(16, iacc22[0])
			if ntypes < 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derred(c3, t)
				}
				if err = ddrvvx(nn, nval, ntypes, dotype, iseed, thresh, nout, a[0], a[1], d[0], d[1], d[2], d[3], a[2], a[3], a[4], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], result, work, lwork, iwork, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "dgeevx", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if c3 == "Dsx" {
			//        ---------------------------------------------------
			//        Dsx:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              dgeesx (Schur form and condition numbers)
			//        ---------------------------------------------------
			fmt.Printf("\n Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    dgeesx (Schur form and condition numbers)\n")
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
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(12, max(11, inmin[0]))
			xlaenv(13, inwin[0])
			xlaenv(14, inibl[0])
			xlaenv(15, ishfts[0])
			xlaenv(16, iacc22[0])
			if ntypes < 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derred(c3, t)
				}
				alareq(ntypes, &dotype)
				if err = ddrvsx(nn, nval, ntypes, dotype, iseed, thresh, nout, a[0], a[1], a[2], d[0], d[1], d[2], d[3], d[4], d[5], a[3], a[4], result, work, lwork, iwork, logwrk, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "dgeesx", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if c3 == "Dgg" {
			//
			//        -------------------------------------------------
			//        Dgg:  Generalized Nonsymmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			if tstchk && tsterr {
				derrgg(c3, t)
			}
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(4, nsval[i-1])
				xlaenv(8, mxbval[i-1])
				xlaenv(16, iacc22[i-1])
				xlaenv(5, nbcol[i-1])
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
					if err = dchkgg(nn, nval, maxtyp, dotype, iseed, thresh, tstdif, thrshn, nout, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], d[0], d[1], d[2], d[3], d[4], d[5], a[12], a[13], work, lwork, logwrk, result, t); err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %4d\n", "dchkgg", info)
					}
				}
			}

		} else if c3 == "Dgs" {
			//        -------------------------------------------------
			//        Dgs:  Generalized Nonsymmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(4, nsval[0])
			xlaenv(8, mxbval[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derrgg(c3, t)
				}
				if err = ddrges(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1], a[2], a[3], a[6], a[7], d[0], d[1], d[2], work, lwork, result, logwrk, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ddrges", info)
				}

				//     Blocked version
				xlaenv(16, 2)
				if err = ddrges3(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1], a[2], a[3], a[6], a[7], d[0], d[1], d[2], work, lwork, result, logwrk, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ddrges3", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if dgx {
			//        -------------------------------------------------
			//        Dgx:  Generalized Nonsymmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(4, nsval[0])
			xlaenv(8, mxbval[0])
			if nn < 0 {
				fmt.Printf(" %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derrgg(c3, t)
				}

				xlaenv(5, 2)
				if err = ddrgsx(nn, ncmax, thresh, nin, nout, a[0], a[1], a[2], a[3], a[4], a[5], d[0], d[1], d[2], c, a[11].VectorIdx(0), work, lwork, iwork, liwork, logwrk, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ddrgsx", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if c3 == "Dgv" {
			//        -------------------------------------------------
			//        Dgv:  Generalized Nonsymmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(4, nsval[0])
			xlaenv(8, mxbval[0])
			if ntypes <= 0 {
				fmt.Printf("\n\n %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derrgg(c3, t)
				}
				if err = ddrgev(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1], a[2], a[3], a[6], a[7], a[8], d[0], d[1], d[2], d[3], d[4], d[5], work, lwork, result, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ddrgev", info)
				}

				//     Blocked version
				if err = ddrgev3(nn, nval, maxtyp, dotype, iseed, thresh, nout, a[0], a[1], a[2], a[3], a[6], a[7], a[8], d[0], d[1], d[2], d[3], d[4], d[5], work, lwork, result, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ddrgev3", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if dxv {
			//        -------------------------------------------------
			//        Dxv:  Generalized Nonsymmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, nbval[0])
			xlaenv(2, nbmin[0])
			xlaenv(3, nxval[0])
			xlaenv(4, nsval[0])
			xlaenv(8, mxbval[0])
			if nn < 0 {
				fmt.Printf(" %3s routines were not tested\n", c3)
			} else {
				if tsterr {
					derrgg(c3, t)
				}

				if err = ddrgvx(nn, thresh, nin, nout, a[0], a[1], a[2], a[3], d[0], d[1], d[2], a[4], a[5], iwork[0], iwork[1], d[3], d[4], d[5], d[6], d[7], d[8], work, lwork, *toSlice(&iwork, 2), liwork-2, result, logwrk, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ddrgvx", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if c3 == "Dsb" {
			//        ------------------------------
			//        Dsb:  Symmetric Band Reduction
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			if tsterr {
				derrst("Dsb", t)
			}
			//         CALL dchksb( NN, NVAL, NK, KVAL, MAXTYP, DOTYPE, ISEED, THRESH,
			//     $                NOUT, A( 1, 1 ), NMAX, D( 1, 1 ), D( 1, 2 ),
			//     $                A( 1, 2 ), NMAX, WORK, LWORK, RESULT, INFO )
			if err = dchksb2stg(nn, nval, nk, kval, maxtyp, dotype, iseed, thresh, nout, a[0], d[0], d[1], d[2], d[3], d[4], a[1], work, lwork, result, t); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "dchksb", info)
			}

		} else if c3 == "Dbb" {
			//        ------------------------------
			//        Dbb:  General Band Reduction
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("\n\n %3s:  NRHS =%4d\n", c3, nrhs)
				if err = dchkbb(nn, mval, nval, nk, kval, maxtyp, dotype, nrhs, &iseed, thresh, nout, a[0], a[1].Off(0, 0).UpdateRows(2*nmax), d[0], d[1], a[3], a[4], a[5], a[6], work, lwork, result, t); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "dchkbb", err)
				}
			}

		} else if c3 == "Glm" {
			//        -----------------------------------------
			//        Glm:  Generalized Linear Regression Model
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			if tsterr {
				derrgg("Glm", t)
			}
			if err = dckglm(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, a[0].Vector(0, 0), a[1].Vector(0, 0), b[0].Vector(0, 0), b[1].Vector(0, 0), x, work, d[0], nout, t); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "dckglm", info)
			}

		} else if c3 == "Gqr" {
			//        ------------------------------------------
			//        Gqr:  Generalized QR and RQ factorizations
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			if tsterr {
				derrgg("Gqr", t)
			}
			if err = dckgqr(nn, mval, nn, pval, nn, nval, ntypes, iseed, thresh, nmax, a[0], a[1], a[2], a[3], taua, b[0], b[1], b[2], b[3], b[4], taub, work, d[0], nout, t); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "dckgqr", info)
			}

		} else if c3 == "Gsv" {
			//        ----------------------------------------------
			//        Gsv:  Generalized Singular Value Decomposition
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			if tsterr {
				derrgg("Gsv", t)
			}
			if err = dckgsv(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, a[0], a[1], b[0], b[1], a[2], b[2], a[3], taua, taub, b[3], iwork, work, d[0], nout, t); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "dckgsv", info)
			}

		} else if c3 == "Csd" {
			//        ----------------------------------------------
			//        Csd:  CS Decomposition
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			if tsterr {
				derrgg("Csd", t)
			}
			if err = dckcsd(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), a[2].VectorIdx(0), a[3].VectorIdx(0), a[4].VectorIdx(0), a[5].VectorIdx(0), a[6].VectorIdx(0), iwork, work, d[0], nout, t); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "dckcsd", info)
			}

		} else if c3 == "Lse" {
			//        --------------------------------------
			//        Lse:  Constrained Linear Least Squares
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
			alareq(ntypes, &dotype)
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			if tsterr {
				derrgg("Lse", t)
			}
			if err = dcklse(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, a[0].VectorIdx(0), a[1].VectorIdx(0), b[0].VectorIdx(0), b[1].VectorIdx(0), x, work, d[0], nout, t); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "dcklse", info)
			}

		} else if c3 == "Dbl" {
			//        DGEBAL:  Balancing
			Dchkbl(t)

		} else if c3 == "Dbk" {
			//        DGEBAK:  Back transformation
			dchkbk(t)

		} else if c3 == "Dgl" {
			//        DGGBAL:  Balancing
			dchkgl(t)

		} else if c3 == "Dgk" {
			//        DGGBAK:  Back transformation
			dchkgk(t)

		} else if c3 == "Dec" {
			//        Dec:  Eigencondition estimation
			xlaenv(1, 1)
			xlaenv(12, 11)
			xlaenv(13, 2)
			xlaenv(14, 0)
			xlaenv(15, 2)
			xlaenv(16, 2)
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
// Nep (Nonsymmetric Eigenvalue Problem):
//     Test ZGEHRD, ZUNGHR, Zhseqr, ZTREVC, ZHSEIN, and ZUNMHR
//
// Sep (Hermitian Eigenvalue Problem):
//     Test ZHETRD, ZUNGTR, ZSTEQR, ZSTERF, ZSTEIN, ZSTEDC,
//     and drivers ZHEEV(X), ZHBEV(X), ZHPEV(X),
//                 ZHEEVD,   ZHBEVD,   ZHPEVD
//
// Svd (Singular Value Decomposition):
//     Test ZGEBRD, ZUNGBR, and ZBDSQR
//     and the drivers ZGESVD, ZGESDD
//
// Zev (Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test Zgeev
//
// Zes (Nonsymmetric Schur form Driver):
//     Test Zgees
//
// Zvx (Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test Zgeevx
//
// Zsx (Nonsymmetric Schur form Expert Driver):
//     Test Zgeesx
//
// Zgg (Generalized Nonsymmetric Eigenvalue Problem):
//     Test ZGGHD3, ZGGBAL, ZGGBAK, ZHGEQZ, and ZTGEVC
//
// Zgs (Generalized Nonsymmetric Schur form Driver):
//     Test Zgges
//
// Zgv (Generalized Nonsymmetric Eigenvalue/eigenvector Driver):
//     Test Zggev
//
// Zgx (Generalized Nonsymmetric Schur form Expert Driver):
//     Test Zggesx
//
// Zxv (Generalized Nonsymmetric Eigenvalue/eigenvector Expert Driver):
//     Test Zggevx
//
// Zsg (Hermitian Generalized Eigenvalue Problem):
//     Test ZHEGST, ZHEGV, ZHEGVD, ZHEGVX, ZHPGST, ZHPGV, ZHPGVD,
//     ZHPGVX, ZHBGST, ZHBGV, ZHBGVD, and ZHBGVX
//
// Zhb (Hermitian Band Eigenvalue Problem):
//     Test Zhbtrd
//
// Zbb (Band Singular Value Decomposition):
//     Test Zgbbrd
//
// Zec (Eigencondition estimation):
//     Test ZTRSYL, ZTREXC, ZTRSNA, and ZTRSEN
//
// Zbl (Balancing a general matrix)
//     Test ZGEBAL
//
// Zbk (Back transformation on a balanced matrix)
//     Test ZGEBAK
//
// Zgl (Balancing a matrix pair)
//     Test ZGGBAL
//
// Zgk (Back transformation on a matrix pair)
//     Test ZGGBAK
//
// Glm (Generalized Linear Regression Model):
//     Tests ZGGGLM
//
// Gqr (Generalized QR and RQ factorizations):
//     Tests ZGGQRF and ZGGRQF
//
// Gsv (Generalized Singular Value Decomposition):
//     Tests ZGGSVD, ZGGSVP, ZTGSJA, ZLAGS2, ZLAPLL, and ZLAPMT
//
// Csd (CS decomposition):
//     Tests ZUNCSD
//
// Lse (Constrained Linear Least Squares):
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
// Zhs or Nep      21     Zchkhs
// Zst or Sep      21     zchkst (routines)
//                 18     zdrvst (drivers)
// Zbd or Svd      16     Zchkbd (routines)
//                  5     ZDRVBD (drivers)
// Zev             21     ZDRVEV
// Zes             21     ZDRVES
// Zvx             21     ZDRVVX
// Zsx             21     ZDRVSX
// Zgg             26     zchkgg (routines)
// Zgs             26     zdrges
// Zgx              5     zdrgsx
// Zgv             26     zdrgev
// Zxv              2     ZDRGVX
// Zsg             21     ZDRVSG
// Zhb             15     ZCHKHB
// Zbb             15     zchkbb
// Zec              -     ZCHKEC
// Zbl              -     ZCHKBL
// Zbk              -     ZCHKBK
// Zgl              -     ZCHKGL
// Zgk              -     ZCHKGK
// Glm              8     zckglm
// Gqr              8     zckgqr
// Gsv              8     zckgsv
// Csd              3     zckcsd
// Lse              8     zcklse
//
//-----------------------------------------------------------------------
//
// Nep input file:
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
// Nep 21
//          requests all of the matrix types for the nonsymmetric
//          eigenvalue problem, while
// Nep  4
// 9 10 11 12
//          requests only matrices of _type 9, 10, 11, and 12.
//
//          The valid 3-character path names are 'Nep' or 'Zhs' for the
//          nonsymmetric eigenvalue routines.
//
//-----------------------------------------------------------------------
//
// Sep or Zsg input file:
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
// lines 13-EOF:  Lines specifying matrix types, as for Nep.
//          The valid 3-character path names are 'Sep' or 'Zst' for the
//          Hermitian eigenvalue routines and driver routines, and
//          'Zsg' for the routines for the Hermitian generalized
//          eigenvalue problem.
//
//-----------------------------------------------------------------------
//
// Svd input file:
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
// lines 15-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path names are 'Svd' or 'Zbd' for both the
//          Svd routines and the Svd driver routines.
//
//-----------------------------------------------------------------------
//
// Zev and Zes data files:
//
// line 1:  'Zev' or 'Zes' in columns 1 to 3.
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
// lines 8 and following:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Zev' to test CGEEV, or
//          'Zes' to test CGEES.
//
//-----------------------------------------------------------------------
//
// The Zvx data has two parts. The first part is identical to Zev,
// and the second part consists of test matrices with precomputed
// solutions.
//
// line 1:  'Zvx' in columns 1-3.
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
// lines 8 and following: The first line contains 'Zvx' in columns 1-3
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
// The Zsx data is like Zvx. The first part is identical to Zev, and the
// second part consists of test matrices with precomputed solutions.
//
// line 1:  'Zsx' in columns 1-3.
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
// lines 8 and following: The first line contains 'Zsx' in columns 1-3
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
// Zgg input file:
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
// lines 17-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Zgg' for the generalized
//          eigenvalue problem routines and driver routines.
//
//-----------------------------------------------------------------------
//
// Zgs and Zgv input files:
//
// line 1:  'Zgs' or 'Zgv' in columns 1 to 3.
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
// lines 7-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Zgs' for the generalized
//          eigenvalue problem routines and driver routines.
//
//-----------------------------------------------------------------------
//
// Zgx input file:
// line 1:  'Zgx' in columns 1 to 3.
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
// Zxv input files:
// line 1:  'Zxv' in columns 1 to 3.
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
// Zhb input file:
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
// lines 8-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Zhb'.
//
//-----------------------------------------------------------------------
//
// Zbb input file:
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
// lines 10-EOF:  Lines specifying matrix types, as for Svd.
//          The 3-character path name is 'Zbb'.
//
//-----------------------------------------------------------------------
//
// Zec input file:
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
// Zbl and Zbk input files:
//
// line 1:  'Zbl' in columns 1-3 to test CGEBAL, or 'Zbk' in
//          columns 1-3 to test CGEBAK.
//
// The remaining lines consist of specially constructed test cases.
//
//-----------------------------------------------------------------------
//
// Zgl and Zgk input files:
//
// line 1:  'Zgl' in columns 1-3 to test ZGGBAL, or 'Zgk' in
//          columns 1-3 to test ZGGBAK.
//
// The remaining lines consist of specially constructed test cases.
//
//-----------------------------------------------------------------------
//
// Glm data file:
//
// line 1:  'Glm' in columns 1 to 3.
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
// lines 9-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Glm' for the generalized
//          linear regression model routines.
//
//-----------------------------------------------------------------------
//
// Gqr data file:
//
// line 1:  'Gqr' in columns 1 to 3.
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
// lines 9-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Gqr' for the generalized
//          QR and RQ routines.
//
//-----------------------------------------------------------------------
//
// Gsv data file:
//
// line 1:  'Gsv' in columns 1 to 3.
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
// lines 9-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Gsv' for the generalized
//          Svd routines.
//
//-----------------------------------------------------------------------
//
// Csd data file:
//
// line 1:  'Csd' in columns 1 to 3.
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
// lines 9-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Csd' for the Csd routine.
//
//-----------------------------------------------------------------------
//
// Lse data file:
//
// line 1:  'Lse' in columns 1 to 3.
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
// lines 9-EOF:  Lines specifying matrix types, as for Nep.
//          The 3-character path name is 'Gsv' for the generalized
//          Svd routines.
//
//-----------------------------------------------------------------------
//
// NMAX is currently set to 132 and must be at least 12 for some of the
// precomputed examples, and LWORK = NMAX*(5*NMAX+20) in the parameter
// statements below.  For Svd, we assume NRHS may be as big as N.  The
// parameter NEED is set to 14 to allow for 14 N-by-N matrices for Zgg.
func TestZeig(t *testing.T) {
	var csd, fatal, glm, gqr, gsv, lse, nep, sep, svd, tstchk, tstdif, tstdrv, tsterr, zbb, zbk, zbl, zes, zev, zgg, zgk, zgl, zgs, zgv, zgx, Zhb, zsx, zvx, zxv bool
	var eps, thresh, thrshn float64
	var i, info, k, liwork, lwork, maxtyp, ncmax, need, newsd, nk, nmax, nn, nparms, nrhs, ntypes, versMajor, versMinor, versPatch int
	var err error

	nmax = 132
	ncmax = 20
	need = 14
	lwork = nmax * (5*nmax + 20)
	liwork = nmax * (nmax + 20)
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
	gltest.Common.Infoc.Errt = fmt.Errorf("")

	versMajor, versMinor, versPatch = golapack.Ilaver()
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
	for _, path := range []string{"Nep", "Sep", "Se2", "Svd", "Zec", "Zes", "Zev", "Zsx", "Zvx", "Zgg", "Zgv", "Zgs", "Zgx", "Zxv", "Zhb", "Zsg", "Zbl", "Zbk", "Zgl", "Zgk", "Zbb", "Glm", "Gqr", "Gsv", "Csd", "Lse"} {
		nep = path == "Nep" || path == "Zhs"
		sep = path == "Sep" || path == "Zst" || path == "Zsg" || path == "Se2"
		svd = path == "Svd" || path == "Zbd"
		zev = path == "Zev"
		zes = path == "Zes"
		zvx = path == "Zvx"
		zsx = path == "Zsx"
		zgg = path == "Zgg"
		zgs = path == "Zgs"
		zgx = path == "Zgx"
		zgv = path == "Zgv"
		zxv = path == "Zxv"
		Zhb = path == "Zhb"
		zbb = path == "Zbb"
		glm = path == "Glm"
		gqr = path == "Gqr" || path == "Grq"
		gsv = path == "Gsv"
		csd = path == "Csd"
		lse = path == "Lse"
		zbl = path == "Zbl"
		zbk = path == "Zbk"
		zgl = path == "Zgl"
		zgk = path == "Zgk"

		//     Report values of parameters.
		fmt.Printf("\n------------------------------------------------------\n")
		if nep {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem routines\n")
		} else if sep {
			fmt.Printf(" Tests of the Hermitian Eigenvalue Problem routines\n")
		} else if svd {
			fmt.Printf(" Tests of the Singular Value Decomposition routines\n")
		} else if zev {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Driver\n    Zgeev (eigenvalues and eigevectors)\n")
		} else if zes {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Driver\n    Zgees (Schur form)\n")
		} else if zvx {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    Zgeevx (eigenvalues, eigenvectors and condition numbers)\n")
		} else if zsx {
			fmt.Printf(" Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n    Zgeesx (Schur form and condition numbers)\n")
		} else if zgg {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem routines\n")
		} else if zgs {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver Zgges\n")
		} else if zgx {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver Zggesx\n")
		} else if zgv {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver Zggev\n")
		} else if zxv {
			fmt.Printf(" Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver Zggevx\n")
		} else if Zhb {
			fmt.Printf(" Tests of Zhbtrd\n (reduction of a Hermitian band matrix to real tridiagonal form)\n")
		} else if zbb {
			fmt.Printf(" Tests of Zgbbrd\n (reduction of a general band matrix to real bidiagonal form)\n")
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
			zchkbl(t)
			continue
		} else if zbk {
			//        ZGEBAK:  Back transformation
			zchkbk(t)
			continue
		} else if zgl {
			//        ZGGBAL:  Balancing
			zchkgl(t)
			continue
		} else if zgk {
			//        ZGGBAK:  Back transformation
			zchkgk(t)
			continue
		} else if path == "Zec" {
			//        Zec:  Eigencondition estimation
			thresh = 20.0
			xlaenv(1, 1)
			xlaenv(12, 1)
			tsterr = true
			zchkec(thresh, tsterr, t)
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

		if path == "Zhs" || path == "Nep" {
			//        -------------------------------------
			//        Nep:  Nonsymmetric Eigenvalue Problem
			//        -------------------------------------
			//        Vary the parameters
			//           NB    = block size
			//           NBMIN = minimum block size
			//           NX    = crossover point
			//           NS    = number of shifts
			//           MAXB  = minimum submatrix size
			// Nep
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
			alareq(ntypes, &dotype)
			xlaenv(1, 1)
			if tsterr {
				zerrhs("Zhseqr", t)
			}
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])
				xlaenv(12, max(11, inmin[i-1]))
				xlaenv(13, inwin[i-1])
				xlaenv(14, inibl[i-1])
				xlaenv(15, ishfts[i-1])
				xlaenv(16, iacc22[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" nb =%4d, nbmin =%4d, nx =%4d, inmin=%4d, inwin =%4d, inibl =%4d, ishfts =%4d, iacc22 =%4d:\n", nbval[i-1], nbmin[i-1], nxval[i-1], max(11, inmin[i-1]), inwin[i-1], inibl[i-1], ishfts[i-1], iacc22[i-1])
				if err = zchkhs(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], a[2], a[3], a[4], a[5], a[6], dc[0], dc[1], a[7], a[8], a[9], a[10], a[11], dc[2], work, lwork, rwork, iwork, logwrk, result); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "Zchkhs", info)
				}
			}

		} else if path == "Zst" || path == "Sep" || path == "Se2" {
			//        ----------------------------------
			//        Sep:  Symmetric Eigenvalue Problem
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
			ntypes = min(maxtyp, ntypes)
			alareq(ntypes, &dotype)
			dotype[8] = false
			fmt.Printf("\n Routines pass computational tests if test ratio is less than%8.2f\n\n", thresh)
			xlaenv(1, 1)
			xlaenv(9, 25)
			if tsterr {
				zerrst("Zst", t)
			}
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf("  nb =%4d, nbmin =%4d, nx =%4d:", nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					if path == "Se2" {
						err = zchkst2stg(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1].CVector(0, 0), dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], dr[6], dr[7], dr[8], dr[9], dr[10], a[2], a[3], a[4].CVector(0, 0), dc[0], a[5], work, lwork, rwork, lwork, iwork, liwork, result)
					} else {
						err = zchkst(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1].CVector(0, 0), dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], dr[6], dr[7], dr[8], dr[9], dr[10], a[2], a[3], a[4].CVector(0, 0), dc[0], a[5], work, lwork, rwork, lwork, iwork, liwork, result)
					}
					if err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %v\n", "zchkst", err)
					}
				}
				if tstdrv {
					if path == "Se2" {
						err = zdrvst2stg(nn, nval, 18, dotype, iseed, thresh, a[0], dr[2], dr[3], dr[4], dr[7], dr[8], dr[9], a[1], a[2], dc[0], a[3], work, lwork, rwork, lwork, iwork, liwork, result)
					} else {
						err = zdrvst(nn, nval, 18, dotype, iseed, thresh, a[0], dr[2], dr[3], dr[4], dr[7], dr[8], dr[9], a[1], a[2], dc[0], a[3], work, lwork, rwork, lwork, iwork, liwork, result)
					}
					if err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %v\n", "zdrvst", err)
					}
				}
			}

		} else if path == "Zsg" {
			//        ----------------------------------------------
			//        Zsg:  Hermitian Generalized Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			xlaenv(9, 25)
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NB =%4d, NBMIN =%4d, NX =%4d\n", path, nbval[i-1], nbmin[i-1], nxval[i-1])
				if tstchk {
					if err = zdrvsg2stg(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], dr[2], dr[3], a[2], a[3], a[4], a[5].CVector(0, 0), a[6].CVector(0, 0), work, lwork, rwork, lwork, iwork, liwork, result); err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %v\n", "ZDRVSG", err)
					}
				}
			}

		} else if path == "Zbd" || path == "Svd" {
			//        ----------------------------------
			//        Svd:  Singular Value Decomposition
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
			alareq(ntypes, &dotype)
			xlaenv(9, 25)

			//        Test the error exits
			xlaenv(1, 1)
			if tsterr && tstchk {
				zerrbd("Zbd", t)
			}
			if tsterr && tstdrv {
				zerred("Zbd", t)
			}

			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(3, nxval[i-1])
				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" nb =%4d, nbmin =%4d, nx =%4d, nrhs =%4d:", nbval[i-1], nbmin[i-1], nxval[i-1], nrhs)
				if tstchk {
					if err = zchkbd(nn, mval, nval, maxtyp, dotype, nrhs, &iseed, thresh, a[0], dr[0], dr[1], dr[2], dr[3], a[1], a[2], a[3], a[4], a[5], a[6], a[7], work, lwork, rwork); err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %v\n", "Zchkbd", err)
					}
				}
				if tstdrv {
					if err = zdrvbd(nn, mval, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], a[2], a[3], a[4], a[5], dr[0], dr[1], dr[2], work, lwork, rwork, iwork); err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %v\n", "Zdrvbd", err)
					}
				}
			}

		} else if path == "Zev" {
			//        --------------------------------------------
			//        Zev:  Nonsymmetric Eigenvalue Problem Driver
			//              Zgeev (eigenvalues and eigenvectors)
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
					zerred(path, t)
				}
				alareq(ntypes, &dotype)
				if err = zdrvev(nn, nval, ntypes, dotype, iseed, thresh, a[0], a[1], dc[0], dc[1], a[2], a[3], a[4], result, work, lwork, rwork, iwork); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "Zgeev", err)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "Zes" {
			//        --------------------------------------------
			//        Zes:  Nonsymmetric Eigenvalue Problem Driver
			//              Zgees (Schur form)
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
					zerred(path, t)
				}
				alareq(ntypes, &dotype)
				if err = zdrves(nn, nval, ntypes, dotype, iseed, thresh, a[0], a[1], a[2], dc[0], dc[1], a[3], result, work, lwork, rwork, iwork, logwrk); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "Zgees", err)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "Zvx" {
			//        --------------------------------------------------------------
			//        Zvx:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              Zgeevx (eigenvalues, eigenvectors and condition numbers)
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
					zerred(path, t)
				}
				alareq(ntypes, &dotype)
				if err = zdrvvx(nn, nval, ntypes, dotype, iseed, thresh, a[0], a[1], dc[0], dc[1], a[2], a[3], a[4], dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], dr[6], dr[7], result, work, lwork, rwork); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "Zgeevx", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "Zsx" {
			//        ---------------------------------------------------
			//        Zsx:  Nonsymmetric Eigenvalue Problem Expert Driver
			//              Zgeesx (Schur form and condition numbers)
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
					zerred(path, t)
				}
				alareq(ntypes, &dotype)
				if err = zdrvsx(nn, nval, ntypes, dotype, iseed, thresh, a[0], a[1], a[2], dc[0], dc[1], dc[2], a[3], a[4], result, work, lwork, rwork, logwrk); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "Zgeesx", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "Zgg" {
			//        -------------------------------------------------
			//        Zgg:  Generalized Nonsymmetric Eigenvalue Problem
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
			alareq(ntypes, &dotype)
			xlaenv(1, 1)
			if tstchk && tsterr {
				zerrgg(path, t)
			}
			for i = 1; i <= nparms; i++ {
				xlaenv(1, nbval[i-1])
				xlaenv(2, nbmin[i-1])
				xlaenv(4, nsval[i-1])
				xlaenv(8, mxbval[i-1])
				xlaenv(16, iacc22[i-1])
				xlaenv(5, nbcol[i-1])

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" nb =%4d, nbmin =%4d, ns =%4d, maxb =%4d, iacc22 =%4d, nbcol =%4d: ", nbval[i-1], nbmin[i-1], nsval[i-1], mxbval[i-1], iacc22[i-1], nbcol[i-1])
				tstdif = false
				thrshn = 10.
				if tstchk {
					if err = zchkgg(nn, nval, maxtyp, dotype, &iseed, thresh, tstdif, thrshn, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], dc[0], dc[1], dc[2], dc[3], a[12], a[13], work, lwork, rwork, logwrk, result); err != nil {
						t.Fail()
						fmt.Printf(" *** Error code from %s = %v\n", "zchkgg", err)
					}
				}
			}

		} else if path == "Zgs" {
			//        -------------------------------------------------
			//        Zgs:  Generalized Nonsymmetric Eigenvalue Problem
			//              Zgges (Schur form)
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
					zerrgg(path, t)
				}
				alareq(ntypes, &dotype)
				if err = zdrges(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], a[2], a[3], a[6], a[7], dc[0], dc[1], work, lwork, rwork, result, logwrk); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "zdrges", err)
				}

				// Blocked version
				if err = zdrges3(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], a[2], a[3], a[6], a[7], dc[0], dc[1], work, lwork, rwork, result, logwrk); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "zdrges3", err)
				}
			}
			fmt.Printf("\n -\n")

		} else if zgx {
			//        -------------------------------------------------
			//        Zgx  Generalized Nonsymmetric Eigenvalue Problem
			//              Zggesx (Schur form and condition numbers)
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
					zerrgg(path, t)
				}
				alareq(ntypes, &dotype)
				xlaenv(5, 2)
				if err = zdrgsx(nn, ncmax, thresh, a[0], a[1], a[2], a[3], a[4], a[5], dc[0], dc[1], c, s, work, lwork, rwork, iwork, liwork, logwrk); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "zdrgsx", err)
				}

				nn = 0
				if err = zdrgsx(nn, ncmax, thresh, a[0], a[1], a[2], a[3], a[4], a[5], dc[0], dc[1], c, s, work, lwork, rwork, iwork, liwork, logwrk); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "zdrgsx", err)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "Zgv" {
			//        -------------------------------------------------
			//        Zgv:  Generalized Nonsymmetric Eigenvalue Problem
			//              Zggev (Eigenvalue/vector form)
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
					zerrgg(path, t)
				}
				alareq(ntypes, &dotype)
				if err = zdrgev(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], a[2], a[3], a[6], a[7], a[8], dc[0], dc[1], dc[2], dc[3], work, lwork, rwork, result); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "zdrgev", err)
				}

				// Blocked version
				xlaenv(16, 2)
				if err = zdrgev3(nn, nval, maxtyp, dotype, iseed, thresh, a[0], a[1], a[2], a[3], a[6], a[7], a[8], dc[0], dc[1], dc[2], dc[3], work, lwork, rwork, result); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "zdrgev3", err)
				}
			}
			fmt.Printf("\n -\n")

		} else if zxv {
			//        -------------------------------------------------
			//        Zxv:  Generalized Nonsymmetric Eigenvalue Problem
			//              Zggevx (eigenvalue/vector with condition numbers)
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
					zerrgg(path, t)
				}
				alareq(ntypes, &dotype)
				if err = zdrgvx(nn, thresh, a[0], a[1], a[2], a[3], dc[0], dc[1], a[4], a[5], iwork[0], iwork[1], dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], work, lwork, rwork, *toSlice(&iwork, 2), *toPtr(liwork - 2), result, logwrk); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %4d\n", "ZDRGVX", info)
				}
			}
			fmt.Printf("\n -\n")

		} else if path == "Zhb" {
			//        ------------------------------
			//        Zhb:  Hermitian Band Reduction
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
			alareq(ntypes, &dotype)
			if newsd == 0 {
				for k = 1; k <= 4; k++ {
					iseed[k-1] = ioldsd[k-1]
				}
			}
			if tsterr {
				zerrst("Zhb", t)
			}
			if err = zchkhb2stg(nn, nval, nk, kval, maxtyp, dotype, iseed, thresh, a[0], dr[0], dr[1], dr[2], dr[3], dr[4], a[1], work, lwork, rwork, result); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %4d\n", "ZCHKHB", info)
			}

		} else if path == "Zbb" {
			//        ------------------------------
			//        Zbb:  General Band Reduction
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
			alareq(ntypes, &dotype)
			for i = 1; i <= nparms; i++ {
				nrhs = nsval[i-1]

				if newsd == 0 {
					for k = 1; k <= 4; k++ {
						iseed[k-1] = ioldsd[k-1]
					}
				}
				fmt.Printf(" %3s:  NRHS =%4d\n", path, nrhs)
				if err = zchkbb(nn, mval, nval, nk, kval, maxtyp, dotype, nrhs, &iseed, thresh, a[0], a[1].Off(0, 0).UpdateRows(2*nmax), dr[0], dr[1], a[3], a[4], a[5], a[6], work, lwork, rwork, result); err != nil {
					t.Fail()
					fmt.Printf(" *** Error code from %s = %v\n", "zchkbb", err)
				}
			}

		} else if path == "Glm" {
			//        -----------------------------------------
			//        Glm:  Generalized Linear Regression Model
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
			xlaenv(1, 1)
			if tsterr {
				zerrgg("Glm", t)
			}
			if err = zckglm(nn, nval, mval, pval, ntypes, iseed, thresh, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), x, work, dr[0]); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %v\n", "zckglm", err)
			}

		} else if path == "Gqr" {
			//        ------------------------------------------
			//        Gqr:  Generalized QR and RQ factorizations
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
			xlaenv(1, 1)
			if tsterr {
				zerrgg("Gqr", t)
			}
			if err = zckgqr(nn, mval, nn, pval, nn, nval, ntypes, iseed, thresh, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), taua, b[0].CVector(0, 0), b[1].CVector(0, 0), b[2].CVector(0, 0), b[3].CVector(0, 0), b[4].CVector(0, 0), taub, work, dr[0]); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %v\n", "zckgqr", err)
			}

		} else if path == "Gsv" {
			//        ----------------------------------------------
			//        Gsv:  Generalized Singular Value Decomposition
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
			xlaenv(1, 1)
			if tsterr {
				zerrgg("Gsv", t)
			}
			if err = zckgsv(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), a[2].CVector(0, 0), b[2].CVector(0, 0), a[3].CVector(0, 0), alpha, beta, b[3].CVector(0, 0), iwork, work, dr[0]); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %v\n", "zckgsv", err)
			}

		} else if path == "Csd" {
			//        ----------------------------------------------
			//        Csd:  CS Decomposition
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
			xlaenv(1, 1)
			if tsterr {
				zerrgg("Csd", t)
			}
			if err = zckcsd(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), a[2].CVector(0, 0), a[3].CVector(0, 0), a[4].CVector(0, 0), a[5].CVector(0, 0), rwork, iwork, work, dr[0]); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %v\n", "zckcsd", err)
			}

		} else if path == "Lse" {
			//        --------------------------------------
			//        Lse:  Constrained Linear Least Squares
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
			xlaenv(1, 1)
			if tsterr {
				zerrgg("Lse", t)
			}
			if err = zcklse(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, a[0].CVector(0, 0), a[1].CVector(0, 0), b[0].CVector(0, 0), b[1].CVector(0, 0), x, work, dr[0]); err != nil {
				t.Fail()
				fmt.Printf(" *** Error code from %s = %v\n", "zcklse", err)
			}

		} else {
			fmt.Printf("\n")
			fmt.Printf(" %3s:  Unrecognized path name\n", path)
		}
	}

	fmt.Printf("\n\n End of tests\n")
}
