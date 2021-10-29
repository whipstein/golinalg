package golapack

import "math"

// Dlags2 computes 2-by-2 orthogonal matrices U, V and Q, such
// that if ( UPPER ) then
//
//           U**T *A*Q = U**T *( A1 A2 )*Q = ( x  0  )
//                             ( 0  A3 )     ( x  x  )
// and
//           V**T*B*Q = V**T *( B1 B2 )*Q = ( x  0  )
//                            ( 0  B3 )     ( x  x  )
//
// or if ( .NOT.UPPER ) then
//
//           U**T *A*Q = U**T *( A1 0  )*Q = ( x  x  )
//                             ( A2 A3 )     ( 0  x  )
// and
//           V**T*B*Q = V**T*( B1 0  )*Q = ( x  x  )
//                           ( B2 B3 )     ( 0  x  )
//
// The rows of the transformed A and B are parallel, where
//
//   U = (  CSU  SNU ), V = (  CSV SNV ), Q = (  CSQ   SNQ )
//       ( -SNU  CSU )      ( -SNV CSV )      ( -SNQ   CSQ )
//
// Z**T denotes the transpose of Z.
func Dlags2(upper bool, a1, a2, a3, b1, b2, b3 float64) (csu, snu, csv, snv, csq, snq float64) {
	var a, aua11, aua12, aua21, aua22, avb11, avb12, avb21, avb22, b, c, csl, csr, d, snl, snr, ua11, ua11r, ua12, ua21, ua22, ua22r, vb11, vb11r, vb12, vb21, vb22, vb22r, zero float64

	zero = 0.0

	if upper {
		//        Input matrices A and B are upper triangular matrices
		//
		//        Form matrix C = A*adj(B) = ( a b )
		//                                   ( 0 d )
		a = a1 * b3
		d = a3 * b1
		b = a2*b1 - a1*b2

		//        The SVD of real 2-by-2 triangular C
		//
		//         ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 )
		//         ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T )
		_, _, snr, csr, snl, csl = Dlasv2(a, b, d)

		if math.Abs(csl) >= math.Abs(snl) || math.Abs(csr) >= math.Abs(snr) {
			//           Compute the (1,1) and (1,2) elements of U**T *A and V**T *B,
			//           and (1,2) element of |U|**T *|A| and |V|**T *|B|.
			ua11r = csl * a1
			ua12 = csl*a2 + snl*a3

			vb11r = csr * b1
			vb12 = csr*b2 + snr*b3

			aua12 = math.Abs(csl)*math.Abs(a2) + math.Abs(snl)*math.Abs(a3)
			avb12 = math.Abs(csr)*math.Abs(b2) + math.Abs(snr)*math.Abs(b3)

			//           zero (1,2) elements of U**T *A and V**T *B
			if (math.Abs(ua11r) + math.Abs(ua12)) != zero {
				if aua12/(math.Abs(ua11r)+math.Abs(ua12)) <= avb12/(math.Abs(vb11r)+math.Abs(vb12)) {
					csq, snq, _ = Dlartg(-ua11r, ua12)
				} else {
					csq, snq, _ = Dlartg(-vb11r, vb12)
				}
			} else {
				csq, snq, _ = Dlartg(-vb11r, vb12)
			}

			csu = csl
			snu = -snl
			csv = csr
			snv = -snr

		} else {
			//           Compute the (2,1) and (2,2) elements of U**T *A and V**T *B,
			//           and (2,2) element of |U|**T *|A| and |V|**T *|B|.
			ua21 = -snl * a1
			ua22 = -snl*a2 + csl*a3

			vb21 = -snr * b1
			vb22 = -snr*b2 + csr*b3

			aua22 = math.Abs(snl)*math.Abs(a2) + math.Abs(csl)*math.Abs(a3)
			avb22 = math.Abs(snr)*math.Abs(b2) + math.Abs(csr)*math.Abs(b3)

			//           zero (2,2) elements of U**T*A and V**T*B, and then swap.
			if (math.Abs(ua21) + math.Abs(ua22)) != zero {
				if aua22/(math.Abs(ua21)+math.Abs(ua22)) <= avb22/(math.Abs(vb21)+math.Abs(vb22)) {
					csq, snq, _ = Dlartg(-ua21, ua22)
				} else {
					csq, snq, _ = Dlartg(-vb21, vb22)
				}
			} else {
				csq, snq, _ = Dlartg(-vb21, vb22)
			}

			csu = snl
			snu = csl
			csv = snr
			snv = csr

		}

	} else {

		//        Input matrices A and B are lower triangular matrices
		//
		//        Form matrix C = A*adj(B) = ( a 0 )
		//                                   ( c d )
		a = a1 * b3
		d = a3 * b1
		c = a2*b3 - a3*b2

		//        The SVD of real 2-by-2 triangular C
		//
		//         ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 )
		//         ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T )
		_, _, snr, csr, snl, csl = Dlasv2(a, c, d)

		if math.Abs(csr) >= math.Abs(snr) || math.Abs(csl) >= math.Abs(snl) {
			//           Compute the (2,1) and (2,2) elements of U**T *A and V**T *B,
			//           and (2,1) element of |U|**T *|A| and |V|**T *|B|.
			ua21 = -snr*a1 + csr*a2
			ua22r = csr * a3

			vb21 = -snl*b1 + csl*b2
			vb22r = csl * b3

			aua21 = math.Abs(snr)*math.Abs(a1) + math.Abs(csr)*math.Abs(a2)
			avb21 = math.Abs(snl)*math.Abs(b1) + math.Abs(csl)*math.Abs(b2)

			//           zero (2,1) elements of U**T *A and V**T *B.
			if (math.Abs(ua21) + math.Abs(ua22r)) != zero {
				if aua21/(math.Abs(ua21)+math.Abs(ua22r)) <= avb21/(math.Abs(vb21)+math.Abs(vb22r)) {
					csq, snq, _ = Dlartg(ua22r, ua21)
				} else {
					csq, snq, _ = Dlartg(vb22r, vb21)
				}
			} else {
				csq, snq, _ = Dlartg(vb22r, vb21)
			}

			csu = csr
			snu = -snr
			csv = csl
			snv = -snl

		} else {
			//           Compute the (1,1) and (1,2) elements of U**T *A and V**T *B,
			//           and (1,1) element of |U|**T *|A| and |V|**T *|B|.
			ua11 = csr*a1 + snr*a2
			ua12 = snr * a3

			vb11 = csl*b1 + snl*b2
			vb12 = snl * b3

			aua11 = math.Abs(csr)*math.Abs(a1) + math.Abs(snr)*math.Abs(a2)
			avb11 = math.Abs(csl)*math.Abs(b1) + math.Abs(snl)*math.Abs(b2)

			//           zero (1,1) elements of U**T*A and V**T*B, and then swap.
			if (math.Abs(ua11) + math.Abs(ua12)) != zero {
				if aua11/(math.Abs(ua11)+math.Abs(ua12)) <= avb11/(math.Abs(vb11)+math.Abs(vb12)) {
					csq, snq, _ = Dlartg(ua12, ua11)
				} else {
					csq, snq, _ = Dlartg(vb12, vb11)
				}
			} else {
				csq, snq, _ = Dlartg(vb12, vb11)
			}

			csu = snr
			snu = csr
			csv = snl
			snv = csl

		}

	}

	return
}
