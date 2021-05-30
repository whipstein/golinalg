package eig

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"math"
)

// Ilaenv returns problem-dependent parameters for the local
// environment.  See ISPEC for a description of the parameters.
//
// In this version, the problem-dependent parameters are contained in
// the integer array IPARMS in the common block CLAENV and the value
// with index ISPEC is copied to ILAENV.  This version of ILAENV is
// to be used in conjunction with XLAENV in TESTING and TIMING.
func Ilaenv(ispec *int, name, opts []byte, n1, n2, n3, n4 *int) (ilaenvReturn int) {
	iparms := &gltest.Common.Claenv.Iparms

	if (*ispec) >= 1 && (*ispec) <= 5 {
		//        Return a value from the common block.
		ilaenvReturn = (*iparms)[(*ispec)-1]

	} else if (*ispec) == 6 {
		//        Compute SVD crossover point.
		ilaenvReturn = int(float64(minint(*n1, *n2)) * 1.6)

	} else if (*ispec) >= 7 && (*ispec) <= 9 {
		//        Return a value from the common block.
		ilaenvReturn = (*iparms)[(*ispec)-1]

	} else if (*ispec) == 10 {
		//        IEEE NaN arithmetic can be trusted not to trap
		//C        ILAENV = 0
		ilaenvReturn = 1
		if ilaenvReturn == 1 {
			ilaenvReturn = golapack.Ieeeck(func() *int { y := 1; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 1.0; return &y }())
		}

	} else if (*ispec) == 11 {
		//        Infinity arithmetic can be trusted not to trap
		//
		//C        ILAENV = 0
		ilaenvReturn = 1
		if ilaenvReturn == 1 {
			ilaenvReturn = golapack.Ieeeck(func() *int { y := 0; return &y }(), func() *float64 { y := 0.0; return &y }(), func() *float64 { y := 1.0; return &y }())
		}

	} else if ((*ispec) >= 12) && ((*ispec) <= 16) {
		//     12 <= ISPEC <= 16: xHSEQR or one of its subroutines.
		ilaenvReturn = (*iparms)[(*ispec)-1]
		//         WRITE(*,*) 'ISPEC = ',ISPEC,' ILAENV =',ILAENV
		//         ILAENV = IPARMQ( ISPEC, NAME, OPTS, N1, N2, N3, N4 )

	} else if ((*ispec) >= 17) && ((*ispec) <= 21) {
		//     17 <= ISPEC <= 21: 2stage eigenvalues SVD routines.
		if (*ispec) == 17 {
			ilaenvReturn = (*iparms)[0]
		} else {
			ilaenvReturn = golapack.Iparam2stage(ispec, name, opts, n1, n2, n3, n4)
		}

	} else {
		//        Invalid value for ISPEC
		ilaenvReturn = -1
	}

	return
}

func Ilaenv2stage(ispec *int, name, opts []byte, n1, n2, n3, n4 *int) (ilaenv2stageReturn int) {
	var iispec int

	iparms := &gltest.Common.Claenv.Iparms

	if ((*ispec) >= 1) && ((*ispec) <= 5) {
		//     1 <= ISPEC <= 5: 2stage eigenvalues SVD routines.
		if (*ispec) == 1 {
			ilaenv2stageReturn = (*iparms)[0]
		} else {
			iispec = 16 + (*ispec)
			ilaenv2stageReturn = golapack.Iparam2stage(&iispec, name, opts, n1, n2, n3, n4)
		}

	} else {
		//        Invalid value for ISPEC
		ilaenv2stageReturn = -1
	}

	return
}

func Iparmq(ispec *int, name, opts []byte, n, ilo, ihi, lwork *int) (iparmqReturn int) {
	var two float64
	var iacc22, inibl, inmin, inwin, ishfts, k22min, kacmin, knwswp, nh, nibble, nmin, ns int

	inmin = 12
	inwin = 13
	inibl = 14
	ishfts = 15
	iacc22 = 16
	nmin = 11
	k22min = 14
	kacmin = 14
	nibble = 14
	knwswp = 500
	two = 2.0

	if ((*ispec) == ishfts) || ((*ispec) == inwin) || ((*ispec) == iacc22) {
		//        ==== Set the number simultaneous shifts ====
		nh = (*ihi) - (*ilo) + 1
		ns = 2
		if nh >= 30 {
			ns = 4
		}
		if nh >= 60 {
			ns = 10
		}
		if nh >= 150 {
			ns = maxint(10, nh/int(math.Round(math.Log(float64(nh))/math.Log(two))))
		}
		if nh >= 590 {
			ns = 64
		}
		if nh >= 3000 {
			ns = 128
		}
		if nh >= 6000 {
			ns = 256
		}
		ns = maxint(2, ns-(ns%2))
	}

	if (*ispec) == inmin {
		//        ===== Matrices of order smaller than NMIN get sent
		//        .     to LAHQR, the classic double shift algorithm.
		//        .     This must be at least 11. ====
		iparmqReturn = nmin

	} else if (*ispec) == inibl {
		//        ==== INIBL: skip a multi-shift qr iteration and
		//        .    whenever aggressive early deflation finds
		//        .    at least (NIBBLE*(window size)/100) deflations. ====
		iparmqReturn = nibble

	} else if (*ispec) == ishfts {
		//        ==== NSHFTS: The number of simultaneous shifts =====
		iparmqReturn = ns

	} else if (*ispec) == inwin {
		//        ==== NW: deflation window size.  ====
		if nh <= knwswp {
			iparmqReturn = ns
		} else {
			iparmqReturn = 3 * ns / 2
		}

	} else if (*ispec) == iacc22 {
		//        ==== IACC22: Whether to accumulate reflections
		//        .     before updating the far-from-diagonal elements
		//        .     and whether to use 2-by-2 block structure while
		//        .     doing it.  A small amount of work could be saved
		//        .     by making this choice dependent also upon the
		//        .     NH=IHI-ILO+1.
		iparmqReturn = 0
		if ns >= kacmin {
			iparmqReturn = 1
		}
		if ns >= k22min {
			iparmqReturn = 2
		}

	} else {
		//        ===== invalid value of ispec =====
		iparmqReturn = -1

	}

	return
}
