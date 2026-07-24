
 add_fsm_encoding \
       {key_detector.state} \
       { }  \
       {{000 000} {001 001} {010 010} {100 011} {101 100} }

 add_fsm_encoding \
       {password_controller.current_state} \
       { }  \
       {{000 000} {001 101} {010 001} {011 010} {100 011} {101 100} {110 110} {111 111} }
